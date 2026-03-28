"""Behavioral observation generators for temporal learning.

Each function generates a sequence of State objects representing what a sensor
module would observe when watching a specific behavior unfold. These model
real physical dynamics as raw sensory feature streams.

The brain learns temporal patterns from sensory data, not object labels.
These generators provide the same kind of raw feature streams that cortical
columns process — location, curvatures, pose orientation, color — changing
over time according to physical dynamics.

No Habitat required. Each behavior is a pure function: physics → State stream.
"""

import numpy as np

from tbp.monty.frameworks.models.states import State


def walking_gait(n_steps=60, gait_period=20, speed=0.005):
    """Bipedal walking observed from the side.

    Models the sensory stream of watching a person walk:
    - Location shifts horizontally at constant speed (forward motion)
    - Curvatures oscillate sinusoidally (knee/hip joint angles bending)
    - Pose vectors oscillate with body sway
    - Color stays constant (same body surface)

    The gait cycle is periodic with period `gait_period` steps.
    """
    states = []
    for t in range(n_steps):
        phase = 2 * np.pi * t / gait_period

        # Forward motion
        location = np.array([t * speed, 0.0, 1.0])

        # Joint curvatures oscillate (knee bends, hip extends)
        knee = np.log1p(0.5 + 0.4 * np.sin(phase))
        hip = np.log1p(0.3 + 0.2 * np.cos(phase))

        # Body sway: slight rotation around vertical axis
        sway = 0.08 * np.sin(phase)
        c, s = np.cos(sway), np.sin(sway)
        pose = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        # Skin/clothing color constant
        hsv = [0.08, 0.55, 0.65]

        states.append(State(
            location=location,
            morphological_features={
                "pose_vectors": pose,
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "hsv": hsv,
                "principal_curvatures_log": [knee, hip],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        ))
    return states


def stapler_press(n_steps=40, press_start=10, press_end=15,
                  release_start=25, release_end=30):
    """Stapler being pressed and released.

    Models top-down observation of a stapler:
    - Open state: high angle between top arm and base, sharp hinge curvature
    - Closing transition: angle decreases smoothly over several steps
    - Closed state: flat, low curvature
    - Opening transition: angle increases back
    - Location fixed (object doesn't move)
    """
    states = []
    for t in range(n_steps):
        location = np.array([0.0, 0.0, 0.1])

        # Angle based on phase
        if t < press_start:
            angle = 0.4  # open (~23 degrees)
        elif t < press_end:
            frac = (t - press_start) / max(press_end - press_start, 1)
            angle = 0.4 * (1 - frac) + 0.05 * frac  # smooth closing
        elif t < release_start:
            angle = 0.05  # closed (nearly flat)
        elif t < release_end:
            frac = (t - release_start) / max(release_end - release_start, 1)
            angle = 0.05 * (1 - frac) + 0.4 * frac  # smooth opening
        else:
            angle = 0.4  # open again

        # Hinge curvature tracks the angle
        hinge_curv = np.log1p(2.0 * angle)
        body_curv = np.log1p(0.05)

        c, s = np.cos(angle), np.sin(angle)
        pose = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Dark metallic color
        hsv = [0.0, 0.0, 0.3]

        states.append(State(
            location=location,
            morphological_features={
                "pose_vectors": pose,
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "hsv": hsv,
                "principal_curvatures_log": [hinge_curv, body_curv],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        ))
    return states


def door_opening(n_steps=40, max_angle=None):
    """Door swinging open on a hinge.

    Monotonic rotation from closed (0 degrees) to open (90 degrees):
    - Location traces an arc (surface point moves as door rotates)
    - Pose rotates around vertical axis
    - Curvatures change as viewing angle changes (foreshortening)
    - Color constant (door surface)
    """
    if max_angle is None:
        max_angle = np.pi / 2

    states = []
    for t in range(n_steps):
        frac = t / max(n_steps - 1, 1)
        angle = max_angle * frac

        # Door surface traces an arc
        radius = 0.4
        location = np.array([
            radius * np.sin(angle),
            0.0,
            radius * np.cos(angle),
        ])

        # Surface normal rotates with door
        c, s = np.cos(angle), np.sin(angle)
        pose = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        # Apparent curvature changes with viewing angle
        curv_1 = np.log1p(0.01 + 0.3 * np.sin(angle))
        curv_2 = np.log1p(0.01)

        # Wood color
        hsv = [0.08, 0.3, 0.6]

        states.append(State(
            location=location,
            morphological_features={
                "pose_vectors": pose,
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "hsv": hsv,
                "principal_curvatures_log": [curv_1, curv_2],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        ))
    return states


def wheel_spinning(n_steps=60, rpm=1.0):
    """Wheel rotating at constant speed.

    Continuous rotation with constant shape:
    - Location fixed (axle center)
    - Pose continuously rotating around one axis
    - Curvatures constant (circular cross-section)
    - Color constant
    """
    states = []
    for t in range(n_steps):
        angle = 2 * np.pi * rpm * t / n_steps

        c, s = np.cos(angle), np.sin(angle)
        pose = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Constant curvature (circle)
        curv = np.log1p(0.5)

        # Dark rubber
        hsv = [0.0, 0.0, 0.4]

        states.append(State(
            location=np.array([0.0, 0.0, 0.0]),
            morphological_features={
                "pose_vectors": pose,
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "hsv": hsv,
                "principal_curvatures_log": [curv, curv],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        ))
    return states


def hand_waving(n_steps=50, wave_period=12):
    """Hand waving gesture.

    Side-to-side oscillation:
    - Location traces a lateral arc with slight vertical bob
    - Pose tilts with the motion direction
    - Curvatures modulate slightly (finger flex)
    - Color constant (skin)
    """
    states = []
    for t in range(n_steps):
        phase = 2 * np.pi * t / wave_period

        # Side-to-side motion with slight vertical bob
        x = 0.15 * np.sin(phase)
        y = 0.03 * np.sin(2 * phase)
        location = np.array([x, y, 0.5])

        # Wrist tilt follows motion direction
        tilt = 0.3 * np.sin(phase)
        c, s = np.cos(tilt), np.sin(tilt)
        pose = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Finger curvatures modulate slightly
        finger_curv = np.log1p(0.6 + 0.2 * np.cos(phase))
        palm_curv = np.log1p(0.1)

        # Skin tone
        hsv = [0.08, 0.5, 0.7]

        states.append(State(
            location=location,
            morphological_features={
                "pose_vectors": pose,
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "hsv": hsv,
                "principal_curvatures_log": [finger_curv, palm_curv],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        ))
    return states


def scissors_cutting(n_steps=50, cut_period=15):
    """Scissors opening and closing cyclically.

    Repetitive open-close cycle for cutting:
    - Location fixed
    - Pose angle oscillates between open and closed
    - Blade curvatures change with angle
    - Color constant (metal)
    """
    states = []
    for t in range(n_steps):
        phase = 2 * np.pi * t / cut_period

        # Blade angle oscillates
        angle = 0.1 + 0.3 * (1 + np.sin(phase)) / 2  # range [0.1, 0.4]

        c, s = np.cos(angle), np.sin(angle)
        pose = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Blade edge curvature changes with opening angle
        blade_curv = np.log1p(0.2 + 0.5 * angle)
        pivot_curv = np.log1p(1.0)  # sharp at pivot

        # Metal color
        hsv = [0.0, 0.1, 0.5]

        states.append(State(
            location=np.array([0.0, 0.0, 0.15]),
            morphological_features={
                "pose_vectors": pose,
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "hsv": hsv,
                "principal_curvatures_log": [blade_curv, pivot_curv],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        ))
    return states


def pendulum_swing(n_steps=80, period=25, damping=0.02):
    """Pendulum swinging with gradual damping.

    Decaying oscillation — tests learning of non-stationary dynamics:
    - Location traces an arc with decreasing amplitude
    - Pose rotation follows the swing
    - Curvatures constant (rigid body)
    - Amplitude decays exponentially
    """
    states = []
    for t in range(n_steps):
        phase = 2 * np.pi * t / period
        amplitude = 0.3 * np.exp(-damping * t)
        angle = amplitude * np.sin(phase)

        # Pendulum bob position
        length = 0.5
        x = length * np.sin(angle)
        y = -length * np.cos(angle)
        location = np.array([x, y, 0.0])

        c, s = np.cos(angle), np.sin(angle)
        pose = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Constant curvature (rigid sphere)
        curv = np.log1p(0.3)

        # Brass color
        hsv = [0.1, 0.7, 0.4]

        states.append(State(
            location=location,
            morphological_features={
                "pose_vectors": pose,
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "hsv": hsv,
                "principal_curvatures_log": [curv, curv],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        ))
    return states


def ball_rolling(n_steps=50, speed=0.01, radius=0.05):
    """Ball rolling along a surface.

    Translation + rotation combined:
    - Location moves linearly (rolling forward)
    - Pose rotates proportional to distance traveled (no slip)
    - Curvatures constant (sphere)
    - Color constant
    """
    states = []
    circumference = 2 * np.pi * radius
    for t in range(n_steps):
        # Linear motion
        x = t * speed
        location = np.array([x, 0.0, radius])

        # Rotation angle from rolling (no slip: angle = distance / radius)
        angle = x / radius

        c, s = np.cos(angle), np.sin(angle)
        pose = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        # Constant sphere curvature
        curv = np.log1p(1.0 / radius)

        # Red ball
        hsv = [0.0, 0.9, 0.8]

        states.append(State(
            location=location,
            morphological_features={
                "pose_vectors": pose,
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "hsv": hsv,
                "principal_curvatures_log": [curv, curv],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        ))
    return states


def light_flickering(n_steps=60, flicker_freq=0.3):
    """Light source flickering (color changes, nothing else moves).

    Tests temporal learning on pure color/brightness changes:
    - Location fixed
    - Pose fixed
    - Curvatures fixed
    - HSV value (brightness) varies stochastically with a bias frequency
    """
    rng = np.random.RandomState(42)
    states = []
    for t in range(n_steps):
        # Brightness oscillates with noise
        base_brightness = 0.5 + 0.3 * np.sin(2 * np.pi * flicker_freq * t)
        brightness = np.clip(base_brightness + 0.1 * rng.randn(), 0, 1)

        states.append(State(
            location=np.array([0.0, 0.0, 2.0]),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "hsv": [0.15, 0.8, brightness],
                "principal_curvatures_log": [np.log1p(0.01), np.log1p(0.01)],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        ))
    return states


def add_noise(states, location_noise=0.001, feature_noise=0.01, seed=None):
    """Add Gaussian noise to a behavior sequence.

    Real sensory data is noisy. This tests robustness of temporal
    learning to observation noise.

    Parameters
    ----------
    states : list[State]
        Clean behavior sequence.
    location_noise : float
        Standard deviation of location noise.
    feature_noise : float
        Standard deviation of feature noise.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    list[State] : noisy version of the sequence.
    """
    rng = np.random.RandomState(seed)
    noisy_states = []

    for state in states:
        loc = state.location.copy() + rng.randn(3) * location_noise

        pv = state.morphological_features["pose_vectors"]
        if hasattr(pv, "copy"):
            pv = pv.copy()
        else:
            pv = np.array(pv, dtype=np.float64)
        pv = pv + rng.randn(3, 3) * feature_noise
        # Re-orthogonalize via Gram-Schmidt
        pv[0] /= np.linalg.norm(pv[0])
        pv[1] -= np.dot(pv[1], pv[0]) * pv[0]
        pv[1] /= np.linalg.norm(pv[1])
        pv[2] = np.cross(pv[0], pv[1])

        nmf = dict(state.non_morphological_features)
        if "hsv" in nmf:
            hsv = np.array(nmf["hsv"]) + rng.randn(3) * feature_noise
            nmf["hsv"] = np.clip(hsv, 0, 1).tolist()
        if "principal_curvatures_log" in nmf:
            curv = np.array(nmf["principal_curvatures_log"])
            curv = curv + rng.randn(len(curv)) * feature_noise
            nmf["principal_curvatures_log"] = curv.tolist()

        noisy_states.append(State(
            location=loc,
            morphological_features={
                "pose_vectors": pv,
                "pose_fully_defined": state.morphological_features.get(
                    "pose_fully_defined", True
                ),
                "on_object": state.morphological_features.get("on_object", 1),
            },
            non_morphological_features=nmf,
            confidence=state.confidence,
            use_state=state.use_state,
            sender_id=state.sender_id,
            sender_type=state.sender_type,
        ))

    return noisy_states
