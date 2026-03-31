"""Quick debug script to check depth values from Panda3D."""
import numpy as np

AGENT_ID = "agent_id_0"
SENSOR_ID = "patch"


def test_at_distance(env, transforms, dist, fov_label=""):
    from tbp.monty.frameworks.sensors import SensorID
    from panda3d.core import LVector3f

    # Reposition camera
    buf = env._sim._agent_buffers[AGENT_ID]
    cam_np = buf["camera_np"]
    cam_np.setPos(0.0, -dist, 0.0)
    cam_np.lookAt(LVector3f(0.0, 0.0, 0.0))

    obs, _ = env._sim.step([])

    raw_depth = obs[AGENT_ID][SensorID(SENSOR_ID)]["depth"].squeeze()
    obj_mask = raw_depth < 9.5
    obj_pixels = obj_mask.sum()
    center_d = float(raw_depth[32, 32])
    print(f"  dist={dist:.2f}m{fov_label}: center_depth={center_d:.3f}, on-obj pixels={obj_pixels}/4096")
    if obj_pixels > 0 and obj_pixels < 200:
        # Show where on-object pixels are
        rows, cols = np.where(obj_mask)
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        r_ctr = int(rows.mean())
        c_ctr = int(cols.mean())
        print(f"    bbox: rows [{r_min},{r_max}] cols [{c_min},{c_max}], centroid=({r_ctr},{c_ctr})")



def main():
    from tbp.monty.simulators.panda3d.environment import Panda3DEnvironment
    from tbp.monty.simulators.panda3d.agents import Panda3DAgent

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

    for dist in [1.5, 0.5, 0.3, 0.2, 0.1]:
        test_at_distance(env, None, dist)

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()

