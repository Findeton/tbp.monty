from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from tbp.monty.hydra import register_resolvers
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.frameworks.run_parallel import (
    generate_parallel_eval_configs,
    generate_parallel_train_configs,
)

SCHEMA_VERSION = 1
DEFAULT_NUM_PARALLEL = 1
DEFAULT_RESOURCE_PROFILE = {
    "num_parallel": DEFAULT_NUM_PARALLEL,
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


@dataclass
class RunManifest:
    schema_version: int
    experiment: str
    overrides: list[str]
    run_mode: str
    run_name: str
    output_dir: str
    log_path: str
    manifest_path: str
    config_path: str
    launch_command: str
    monitor_command: str
    resume_command: str
    stop_command: str
    summary_command: str
    expected_artifacts: list[str]
    resource_profile: dict[str, str | int]
    created_at: str
    git_commit: str | None
    notes: str = ""
    pid: int | None = None
    launched_at: str | None = None
    last_resume_at: str | None = None
    status: str = "planned"
    resume_history: list[str] = field(default_factory=list)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def config_root() -> Path:
    return repo_root() / "src" / "tbp" / "monty" / "conf"


def clear_hydra_state() -> None:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def compose_experiment(experiment: str, overrides: list[str] | None = None) -> DictConfig:
    setup_env()
    register_resolvers()
    clear_hydra_state()
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(config_root()),
    ):
        return hydra.compose(
            config_name="experiment",
            overrides=[f"experiment={experiment}", *(overrides or [])],
        )


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def format_episode_spec(indices: list[int]) -> str:
    if not indices:
        return ""

    ranges: list[str] = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue

        ranges.append(_format_range(start, prev))
        start = idx
        prev = idx

    ranges.append(_format_range(start, prev))
    return ",".join(ranges)


def _format_range(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}:{end + 1}"


def _determine_run_mode(cfg: DictConfig) -> str:
    do_train = bool(cfg.experiment.config.do_train)
    do_eval = bool(cfg.experiment.config.do_eval)
    if do_train and not do_eval:
        return "parallel_train"
    if do_eval and not do_train:
        return "parallel_eval"
    raise ValueError(
        "Phase 0 run management currently supports either training-only or "
        "evaluation-only run_parallel workflows."
    )


def _is_pretraining(cfg: DictConfig) -> bool:
    target = str(cfg.experiment.get("_target_", ""))
    return target.endswith(
        "pretraining_experiments.MontySupervisedObjectPretrainingExperiment"
    )


def _base_output_dir(cfg: DictConfig) -> Path:
    output_dir = Path(str(cfg.experiment.config.logging.output_dir)).expanduser()
    run_name = str(cfg.experiment.config.logging.run_name)
    return output_dir / run_name


def _has_csv_stats_handler(cfg: DictConfig) -> bool:
    handlers = cfg.experiment.config.logging.get("monty_handlers", [])
    return any(issubclass(handler, BasicCSVStatsHandler) for handler in handlers)


def _expected_artifacts(cfg: DictConfig) -> list[Path]:
    base_dir = _base_output_dir(cfg)
    run_mode = _determine_run_mode(cfg)
    artifacts: list[Path] = [base_dir / "phase0_manifest.json", base_dir / "phase0_config.yaml"]
    if run_mode == "parallel_train":
        if _is_pretraining(cfg):
            artifacts.append(base_dir / "pretrained" / "model.pt")
        else:
            artifacts.append(base_dir / "model.pt")
        if _has_csv_stats_handler(cfg):
            artifacts.append(base_dir / "train_stats.csv")
    else:
        if _has_csv_stats_handler(cfg):
            artifacts.append(base_dir / "eval_stats.csv")
    artifacts.append(base_dir / "parallel_log.txt")
    artifacts.append(base_dir / "phase0_run.log")
    return artifacts


def _parallel_episode_dirs(cfg: DictConfig) -> list[Path]:
    base_dir = _base_output_dir(cfg)
    run_name = str(cfg.experiment.config.logging.run_name)
    run_mode = _determine_run_mode(cfg)
    pattern = f"{run_name}-{run_mode}_episode_*"
    return sorted(base_dir.glob(pattern))


def _episode_configs(cfg: DictConfig) -> list[dict[str, Any]]:
    name = str(cfg.experiment.config.logging.run_name)
    run_mode = _determine_run_mode(cfg)
    if run_mode == "parallel_train":
        return list(generate_parallel_train_configs(cfg.experiment, name))
    return list(generate_parallel_eval_configs(cfg.experiment, name))


def _episode_artifact(config: dict[str, Any], pretraining: bool) -> Path:
    output_dir = Path(config["config"]["logging"]["output_dir"])
    if pretraining:
        return output_dir / "pretrained" / "model.pt"
    if "parallel_eval_episode_" in str(output_dir):
        return output_dir / "eval_stats.csv"
    return output_dir / "model.pt"


def _completed_episode_indices(cfg: DictConfig) -> list[int]:
    configs = _episode_configs(cfg)
    pretraining = _is_pretraining(cfg)
    completed = []
    for idx, config in enumerate(configs):
        if _episode_artifact(config, pretraining).exists():
            completed.append(idx)
    return completed


def _expected_completion(cfg: DictConfig) -> dict[str, Any]:
    configs = _episode_configs(cfg)
    if _outputs_complete(cfg):
        completed = list(range(len(configs)))
        return {
            "total": len(configs),
            "completed": completed,
            "missing": [],
            "missing_spec": "",
        }

    completed = _completed_episode_indices(cfg)
    missing = [idx for idx in range(len(configs)) if idx not in completed]
    return {
        "total": len(configs),
        "completed": completed,
        "missing": missing,
        "missing_spec": format_episode_spec(missing),
    }


def _outputs_complete(cfg: DictConfig) -> bool:
    expected = _expected_artifacts(cfg)
    if _parallel_episode_dirs(cfg):
        return False
    return all(path.exists() for path in expected if path.name not in {"phase0_manifest.json", "phase0_config.yaml", "phase0_run.log"})


def _is_process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _resource_profile(num_parallel: int) -> dict[str, str | int]:
    profile = dict(DEFAULT_RESOURCE_PROFILE)
    profile["num_parallel"] = num_parallel
    return profile


def _launch_parts(experiment: str, overrides: list[str], num_parallel: int, episodes: str | None = None) -> list[str]:
    parts = [
        sys.executable,
        "run_parallel.py",
        f"experiment={experiment}",
        f"num_parallel={num_parallel}",
    ]
    if episodes:
        parts.append(f"episodes='{episodes}'")
    parts.extend(overrides)
    return parts


def _chat_commands(base_dir: Path, experiment: str, overrides: list[str], num_parallel: int) -> dict[str, str]:
    manifest_path = base_dir / "phase0_manifest.json"
    launch = quote_command(_launch_parts(experiment, overrides, num_parallel))
    tool = Path("tools") / "phase0_runs.py"
    monitor = quote_command([sys.executable, str(tool), "status", "--manifest", str(manifest_path)])
    resume = quote_command([sys.executable, str(tool), "resume", "--manifest", str(manifest_path)])
    stop = quote_command([sys.executable, str(tool), "stop", "--manifest", str(manifest_path)])
    summary = quote_command([sys.executable, str(tool), "summarize", "--manifest", str(manifest_path)])
    return {
        "launch": launch,
        "monitor": monitor,
        "resume": resume,
        "stop": stop,
        "summary": summary,
    }


def build_manifest(experiment: str, overrides: list[str], num_parallel: int) -> RunManifest:
    cfg = compose_experiment(experiment, overrides)
    base_dir = _base_output_dir(cfg)
    base_dir.mkdir(parents=True, exist_ok=True)
    commands = _chat_commands(base_dir, experiment, overrides, num_parallel)
    manifest = RunManifest(
        schema_version=SCHEMA_VERSION,
        experiment=experiment,
        overrides=list(overrides),
        run_mode=_determine_run_mode(cfg),
        run_name=str(cfg.experiment.config.logging.run_name),
        output_dir=str(base_dir),
        log_path=str(base_dir / "phase0_run.log"),
        manifest_path=str(base_dir / "phase0_manifest.json"),
        config_path=str(base_dir / "phase0_config.yaml"),
        launch_command=commands["launch"],
        monitor_command=commands["monitor"],
        resume_command=commands["resume"],
        stop_command=commands["stop"],
        summary_command=commands["summary"],
        expected_artifacts=[str(path) for path in _expected_artifacts(cfg)],
        resource_profile=_resource_profile(num_parallel),
        created_at=utcnow(),
        git_commit=get_git_commit(),
    )
    _write_manifest_files(manifest, cfg)
    return manifest


def _write_manifest_files(manifest: RunManifest, cfg: DictConfig) -> None:
    manifest_path = Path(manifest.manifest_path)
    config_path = Path(manifest.config_path)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
    config_path.write_text(OmegaConf.to_yaml(cfg))
    _write_helper_script(Path(manifest.output_dir) / "phase0_launch.sh", manifest.launch_command)
    _write_helper_script(Path(manifest.output_dir) / "phase0_status.sh", manifest.monitor_command)
    _write_helper_script(Path(manifest.output_dir) / "phase0_resume.sh", manifest.resume_command)
    _write_helper_script(Path(manifest.output_dir) / "phase0_stop.sh", manifest.stop_command)
    _write_helper_script(Path(manifest.output_dir) / "phase0_summary.sh", manifest.summary_command)


def _write_helper_script(path: Path, command: str) -> None:
    path.write_text(f"#!/bin/sh\nset -eu\ncd {shlex.quote(str(repo_root()))}\n{command}\n")


def load_manifest(path: Path) -> RunManifest:
    data = json.loads(path.read_text())
    return RunManifest(**data)


def save_manifest(manifest: RunManifest) -> None:
    Path(manifest.manifest_path).write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
    )


def load_cfg_from_manifest(manifest: RunManifest) -> DictConfig:
    return compose_experiment(manifest.experiment, manifest.overrides)


def get_status(manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    cfg = load_cfg_from_manifest(manifest)
    completion = _expected_completion(cfg)
    outputs_complete = _outputs_complete(cfg)
    pid_alive = _is_process_alive(manifest.pid)
    if outputs_complete:
        state = "completed"
    elif pid_alive:
        state = "running"
    elif completion["completed"]:
        state = "partial"
    else:
        state = manifest.status
    return {
        "state": state,
        "pid": manifest.pid,
        "pid_alive": pid_alive,
        "output_dir": manifest.output_dir,
        "run_name": manifest.run_name,
        "mode": manifest.run_mode,
        "completed_episodes": completion["completed"],
        "missing_episodes": completion["missing"],
        "missing_spec": completion["missing_spec"],
        "expected_artifacts": manifest.expected_artifacts,
        "outputs_complete": outputs_complete,
        "monitor_command": manifest.monitor_command,
        "resume_command": manifest.resume_command,
        "summary_command": manifest.summary_command,
    }


def launch(manifest_path: Path, background: bool = True, episodes: str | None = None) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    if _is_process_alive(manifest.pid):
        raise RuntimeError(f"Run is already active with PID {manifest.pid}.")

    env = os.environ.copy()
    for key, value in manifest.resource_profile.items():
        if key == "num_parallel":
            continue
        env[key] = str(value)

    num_parallel = int(manifest.resource_profile["num_parallel"])
    parts = _launch_parts(manifest.experiment, manifest.overrides, num_parallel, episodes)
    log_path = Path(manifest.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if background:
        with log_path.open("ab") as log_file:
            proc = subprocess.Popen(
                parts,
                cwd=repo_root(),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        manifest.pid = proc.pid
        if episodes is not None:
            manifest.last_resume_at = utcnow()
            manifest.resume_history.append(episodes)
        else:
            manifest.launched_at = utcnow()
        manifest.status = "running"
        save_manifest(manifest)
        return {
            "pid": proc.pid,
            "log_path": str(log_path),
            "command": quote_command(parts),
        }

    result = subprocess.run(parts, cwd=repo_root(), env=env, check=False)
    manifest.status = "completed" if result.returncode == 0 else "failed"
    save_manifest(manifest)
    return {"returncode": result.returncode, "command": quote_command(parts)}


def resume(manifest_path: Path, background: bool = True) -> dict[str, Any]:
    status = get_status(manifest_path)
    if status["state"] == "completed":
        return {"message": "Run already completed.", "status": status}
    missing_spec = status["missing_spec"]
    if not missing_spec:
        return {"message": "No missing episodes detected.", "status": status}
    return launch(manifest_path, background=background, episodes=missing_spec)


def stop(manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    if not _is_process_alive(manifest.pid):
        manifest.status = "stopped"
        manifest.pid = None
        save_manifest(manifest)
        return {"message": "No active process found."}

    assert manifest.pid is not None
    os.killpg(manifest.pid, signal.SIGTERM)
    manifest.status = "stopped"
    manifest.pid = None
    save_manifest(manifest)
    return {"message": "Stop signal sent."}


def summarize(manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    cfg = load_cfg_from_manifest(manifest)
    base_dir = Path(manifest.output_dir)
    summary: dict[str, Any] = {
        "run_name": manifest.run_name,
        "run_mode": manifest.run_mode,
        "output_dir": manifest.output_dir,
        "status": get_status(manifest_path)["state"],
    }
    if manifest.run_mode == "parallel_train":
        train_csv = base_dir / "train_stats.csv"
        model_path = base_dir / ("pretrained/model.pt" if _is_pretraining(cfg) else "model.pt")
        summary["train_stats_exists"] = train_csv.exists()
        summary["model_exists"] = model_path.exists()
        if train_csv.exists():
            train_stats = pd.read_csv(train_csv)
            summary["train_rows"] = len(train_stats)
            summary["train_columns"] = list(train_stats.columns)
        if model_path.exists():
            state = torch.load(model_path, map_location="cpu")
            summary["lm_graph_counts"] = {
                str(lm): len(state["lm_dict"][lm]["graph_memory"])
                for lm in state["lm_dict"]
            }
    else:
        eval_csv = base_dir / "eval_stats.csv"
        summary["eval_stats_exists"] = eval_csv.exists()
        if eval_csv.exists():
            eval_stats = pd.read_csv(eval_csv)
            summary["eval_rows"] = len(eval_stats)
            if len(eval_stats) > 0 and "primary_performance" in eval_stats.columns:
                perf = eval_stats["primary_performance"]
                summary["percent_correct"] = float(
                    ((perf == "correct") | (perf == "correct_mlh")).mean() * 100
                )
                summary["performance_counts"] = perf.value_counts().to_dict()
    summary_path = Path(manifest.output_dir) / "phase0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def print_pretty(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 0 run management for Monty benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--experiment", required=True)
    prepare_parser.add_argument("--override", action="append", default=[])
    prepare_parser.add_argument("--num-parallel", type=int, default=DEFAULT_NUM_PARALLEL)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--manifest", type=Path)
    launch_parser.add_argument("--experiment")
    launch_parser.add_argument("--override", action="append", default=[])
    launch_parser.add_argument("--num-parallel", type=int, default=DEFAULT_NUM_PARALLEL)
    launch_parser.add_argument("--foreground", action="store_true")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--manifest", type=Path, required=True)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--manifest", type=Path, required=True)
    resume_parser.add_argument("--foreground", action="store_true")

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--manifest", type=Path, required=True)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--manifest", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        manifest = build_manifest(args.experiment, args.override, args.num_parallel)
        print_pretty(asdict(manifest))
        return 0

    if args.command == "launch":
        if args.manifest is None:
            if not args.experiment:
                parser.error("launch requires either --manifest or --experiment")
            manifest = build_manifest(args.experiment, args.override, args.num_parallel)
            manifest_path = Path(manifest.manifest_path)
        else:
            manifest_path = args.manifest
        print_pretty(launch(manifest_path, background=not args.foreground))
        return 0

    if args.command == "status":
        print_pretty(get_status(args.manifest))
        return 0

    if args.command == "resume":
        print_pretty(resume(args.manifest, background=not args.foreground))
        return 0

    if args.command == "stop":
        print_pretty(stop(args.manifest))
        return 0

    if args.command == "summarize":
        print_pretty(summarize(args.manifest))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())