"""Quick debug script - iterative centering from 0.3m without initial lookAt."""
import numpy as np

AGENT_ID = "agent_id_0"
SENSOR_ID = "patch"
BG = 9.5


def main():
    from tbp.monty.simulators.panda3d.environment import Panda3DEnvironment
    from tbp.monty.simulators.panda3d.agents import Panda3DAgent
    from tbp.monty.frameworks.sensors import SensorID

    agent = Panda3DAgent(
        agent_id=AGENT_ID,
        sensor_id=SENSOR_ID,
        position=(0.0, -0.3, 0.0),  # Start at 0.3m, no lookAt
        rotation=(1.0, 0.0, 0.0, 0.0),
        resolution=(64, 64),
        fov=90.0,
        action_space_type="distant_agent",
    )

    env = Panda3DEnvironment(
        agents=[agent],
        near=0.01,
        far=10.0,
        object_registry={
            "011_banana": {
                "model_path": "/Users/felixrobles/tbp/data/habitat/objects/ycb/meshes/011_banana/google_16k/textured.glb.orig",
                "animated": False,
            }
        },
    )

    env.add_object("011_banana", position=(0.0, 0.0, 0.0))
    env.reset()

    sim = env._sim
    buf = sim._agent_buffers[AGENT_ID]
    cam_np = buf["camera_np"]
    h, w = 64, 64
    fov = 90.0
    px_per_deg = w / fov

    print("Iterative centering from 0.3m (NO initial lookAt):")
    for it in range(6):
        obs, _ = sim.step([])
        depth = obs[AGENT_ID][SensorID(SENSOR_ID)]["depth"].squeeze()
        on_obj = depth < BG
        n = on_obj.sum()
        center_on = depth[h // 2, w // 2] < BG
        rows, cols = np.where(on_obj) if n > 0 else (np.array([]), np.array([]))
        cen_r = rows.mean() if n > 0 else -1
        cen_c = cols.mean() if n > 0 else -1
        print(f"  iter {it}: n_obj={n}, center_on={center_on}, centroid=({cen_r:.1f},{cen_c:.1f})")

        if center_on or n == 0:
            break

        d_col = cen_c - (w / 2)
        d_row = cen_r - (h / 2)
        if abs(d_col) < 0.5 and abs(d_row) < 0.5:
            break

        # Sign convention:
        # d_col < 0 → object is LEFT → turn camera LEFT → positive heading in P3D
        # d_row < 0 → object is ABOVE → look UP → positive pitch in P3D
        delta_h = -d_col / px_per_deg   # negate: left error → positive heading
        delta_p = -d_row / px_per_deg   # negate: above error → positive pitch
        hpr = cam_np.getHpr()
        cam_np.setH(hpr[0] + delta_h)
        cam_np.setP(hpr[1] + delta_p)
        print(f"    → adjusted H by {delta_h:.2f}°, P by {delta_p:.2f}°")

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()

import numpy as np

AGENT_ID = "agent_id_0"
SENSOR_ID = "patch"
BG = 9.5


def main():
    from tbp.monty.simulators.panda3d.environment import Panda3DEnvironment
    from tbp.monty.simulators.panda3d.agents import Panda3DAgent
    from tbp.monty.frameworks.sensors import SensorID
    from panda3d.core import LVector3f

    agent = Panda3DAgent(
        agent_id=AGENT_ID,
        sensor_id=SENSOR_ID,
        position=(0.0, -0.1, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
        resolution=(64, 64),
        fov=90.0,
        action_space_type="distant_agent",
    )

    env = Panda3DEnvironment(
        agents=[agent],
        near=0.01,
        far=10.0,
        object_registry={
            "011_banana": {
                "model_path": "/Users/felixrobles/tbp/data/habitat/objects/ycb/meshes/011_banana/google_16k/textured.glb.orig",
                "animated": False,
            }
        },
    )

    env.add_object("011_banana", position=(0.0, 0.0, 0.0))
    env.reset()

    buf = env._sim._agent_buffers[AGENT_ID]
    cam_np = buf["camera_np"]

    # Find object center from bounding box
    for obj_np in env._sim._objects.values():
        bounds = obj_np.getTightBounds()
        if bounds:
            lo, hi = bounds
            cx = (lo[0] + hi[0]) / 2
            cy = (lo[1] + hi[1]) / 2
            cz = (lo[2] + hi[2]) / 2
            print(f"Object center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")

    for dist in [0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05]:
        cam_np.setPos(0.0, -dist, 0.0)
        cam_np.lookAt(LVector3f(0.011, -0.018, -0.007))  # Bounding box center
        obs, _ = env._sim.step([])
        depth = obs[AGENT_ID][SensorID(SENSOR_ID)]["depth"].squeeze()
        on_obj = depth < BG
        n = on_obj.sum()
        center_on = depth[32, 32] < BG

        if n > 0:
            rows, cols = np.where(on_obj)
            cen_row, cen_col = rows.mean(), cols.mean()
        else:
            cen_row, cen_col = -1, -1

        print(f"dist={dist:.2f}m: center_on={center_on}, n_obj={n}, centroid=({cen_row:.1f},{cen_col:.1f})")

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()

import numpy as np

AGENT_ID = "agent_id_0"
SENSOR_ID = "patch"
BG = 9.5


def center_camera_on_object(env, max_iters=5):
    """Iteratively adjust heading/pitch to center object pixels at image center."""
    sim = env._sim
    for agent in sim._agents:
        buf = sim._agent_buffers[agent.agent_id]
        cam_np = buf["camera_np"]
        h, w = agent.resolution
        px_per_deg = w / agent.fov

        for iteration in range(max_iters):
            obs, _ = sim.step([])
            from tbp.monty.frameworks.sensors import SensorID
            depth = obs[agent.agent_id][SensorID(agent.sensor_id)]["depth"].squeeze()
            on_obj = depth < BG
            n_obj = on_obj.sum()

            center_on = depth[h // 2, w // 2] < BG
            print(f"  iter {iteration}: n_obj={n_obj}, center_on={center_on}")

            if n_obj == 0 or center_on:
                break

            rows, cols = np.where(on_obj)
            r_ctr = rows.mean()
            c_ctr = cols.mean()
            d_col = c_ctr - (w / 2)
            d_row = r_ctr - (h / 2)

            if abs(d_col) < 0.5 and abs(d_row) < 0.5:
                break

            delta_h = d_col / px_per_deg
            delta_p = d_row / px_per_deg
            hpr = cam_np.getHpr()
            cam_np.setH(hpr[0] + delta_h)
            cam_np.setP(hpr[1] + delta_p)

    return obs


def main():
    from tbp.monty.simulators.panda3d.environment import Panda3DEnvironment
    from tbp.monty.simulators.panda3d.agents import Panda3DAgent
    from tbp.monty.frameworks.sensors import SensorID

    agent = Panda3DAgent(
        agent_id=AGENT_ID,
        sensor_id=SENSOR_ID,
        position=(0.0, -1.5, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
        resolution=(64, 64),
        fov=90.0,
        action_space_type="distant_agent",
    )

    env = Panda3DEnvironment(
        agents=[agent],
        near=0.01,
        far=10.0,
        object_registry={
            "011_banana": {
                "model_path": "/Users/felixrobles/tbp/data/habitat/objects/ycb/meshes/011_banana/google_16k/textured.glb.orig",
                "animated": False,
            }
        },
    )

    env.add_object("011_banana", position=(0.0, 0.0, 0.0))
    env.reset()

    print("Testing iterative centering at dist=1.5m:")
    obs = center_camera_on_object(env, max_iters=5)

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()

import numpy as np

AGENT_ID = "agent_id_0"
SENSOR_ID = "patch"


def look_at_scene_center(env):
    from panda3d.core import LVector3f

    sim = env._sim
    centers = []
    for obj_np in sim._objects.values():
        bounds = obj_np.getTightBounds()
        if bounds:
            lo, hi = bounds
            centers.append((
                (lo[0] + hi[0]) / 2,
                (lo[1] + hi[1]) / 2,
                (lo[2] + hi[2]) / 2,
            ))
    if not centers:
        print("No objects in scene!")
        return

    cx = sum(c[0] for c in centers) / len(centers)
    cy = sum(c[1] for c in centers) / len(centers)
    cz = sum(c[2] for c in centers) / len(centers)
    target = LVector3f(float(cx), float(cy), float(cz))
    print(f"Scene center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")

    for agent in sim._agents:
        buf = sim._agent_buffers[agent.agent_id]
        cam_np = buf["camera_np"]
        cam_np.lookAt(target)


def main():
    from tbp.monty.simulators.panda3d.environment import Panda3DEnvironment
    from tbp.monty.simulators.panda3d.agents import Panda3DAgent
    from tbp.monty.frameworks.sensors import SensorID

    agent = Panda3DAgent(
        agent_id=AGENT_ID,
        sensor_id=SENSOR_ID,
        position=(0.0, -1.5, 0.0),  # -Y: looks along +Y toward origin
        rotation=(1.0, 0.0, 0.0, 0.0),
        resolution=(64, 64),
        fov=90.0,
        action_space_type="distant_agent",
    )

    env = Panda3DEnvironment(
        agents=[agent],
        near=0.01,
        far=10.0,
        object_registry={
            "011_banana": {
                "model_path": "/Users/felixrobles/tbp/data/habitat/objects/ycb/meshes/011_banana/google_16k/textured.glb.orig",
                "animated": False,
            }
        },
    )

    env.add_object("011_banana", position=(0.0, 0.0, 0.0))
    env.reset()

    from panda3d.core import LVector3f

    buf = env._sim._agent_buffers[AGENT_ID]
    cam_np = buf["camera_np"]
    cam_np.setPos(0.0, -0.3, 0.0)

    print(f"Before lookAt: HPR={cam_np.getHpr()}, pos={cam_np.getPos()}")

    # Find object bounding box
    for obj_id, obj_np in env._sim._objects.items():
        bounds = obj_np.getTightBounds()
        if bounds:
            lo, hi = bounds
            print(f"Object tight bounds in world: lo={lo}, hi={hi}")
            cx = (lo[0] + hi[0]) / 2
            cy = (lo[1] + hi[1]) / 2
            cz = (lo[2] + hi[2]) / 2
            print(f"Bounding box center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
            # Look at the actual bounding box center
            cam_np.lookAt(LVector3f(float(cx), float(cy), float(cz)))
        else:
            print("No bounds!")

    print(f"After lookAt: HPR={cam_np.getHpr()}, pos={cam_np.getPos()}")

    obs, _ = env._sim.step([])
    raw_depth = obs[AGENT_ID][SensorID(SENSOR_ID)]["depth"].squeeze()
    obj_mask = raw_depth < 9.5
    rows, cols = np.where(obj_mask)
    if len(rows) > 0:
        print(f"Object pixels: {len(rows)}, rows=[{rows.min()},{rows.max()}] cols=[{cols.min()},{cols.max()}]")
        print(f"Centroid: ({rows.mean():.1f}, {cols.mean():.1f})")
        print(f"Center (32,32) depth: {raw_depth[32,32]:.3f}")
        print(f"Depth map around center (30:35, 27:37):")
        print(raw_depth[28:38, 27:38].round(2))

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
