#!/usr/bin/env python

from tbp.monty.frameworks.run_env import setup_env

setup_env()

from tbp.monty.frameworks.utils.phase0_runs import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())