import math
import logging

# ------------- logging config --------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all levels

file_handler = logging.FileHandler('relay_receiver_debug.log')
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ------------- helper geometry functions --------------------

def get_keypoint(kpts, idx):
    """Retrieve keypoint by index."""
    if idx < len(kpts):
        return kpts[idx]
    return None

def distance_2d(p1, p2):
    """Calculate Euclidean distance between two 2D points."""
    if p1 is None or p2 is None:
        return None
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def compute_angle_3pts(a, b, c):
    """
    Compute the angle (in degrees) formed at point b by points a, b, and c.
    Returns None if any point is missing or if the angle can't be computed.
    """
    if a is None or b is None or c is None:
        return None
    ax, ay = a
    bx, by = b
    cx, cy = c
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 < 1e-5 or mag2 < 1e-5:
        return None
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    try:
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    except ValueError:
        logger.debug(f"Invalid angle calculation with cos_angle={cos_angle}")
        return None

# ------------- phase detection and segmentation --------------------

def detect_phase_transitions(player_coords):
    """
    Detect simple phase boundaries by dividing frames into thirds.
    Customize this to real cues (like runner crossing a line).
    """
    if not player_coords:
        logger.debug("player_coords is empty.")
        return 0, 0

    total = len(player_coords)
    one_third = total // 3
    two_thirds = (2 * total) // 3

    logger.debug(f"one_third={one_third}, two_thirds={two_thirds}")

    return one_third, two_thirds

def segment_video_into_phases(player_coords, a_end_index, b_end_index):
    """
    Segment frames into Phase A, Phase B, and Phase C.
    """
    phase_a_frames = player_coords[:a_end_index]
    phase_b_frames = player_coords[a_end_index:b_end_index]
    phase_c_frames = player_coords[b_end_index:]

    logger.debug(
        f"Segments: "
        f"phaseA_frames={len(phase_a_frames)}, "
        f"phaseB_frames={len(phase_b_frames)}, "
        f"phaseC_frames={len(phase_c_frames)}"
    )
    return phase_a_frames, phase_b_frames, phase_c_frames

# ------------- criterion checks by phase --------------------

def evaluate_phase_a(phase_a_frames):
    """
    Phase A (Wait/Approach):
    - Criterion 1: 
      "Receiver leaves after runner passes marker. He runs without looking back or pulling hand"
    """
    partial_scoring = {
        'departures_after_marker': 0
    }
    partial_eval_frames = {
        1: []
    }

    logger.debug(f"PHASE=A: Checking Criterion 1 over {len(phase_a_frames)} frames.")

    for idx, data in enumerate(phase_a_frames):
        frame = data.get('frame', idx)
        kpts = data['keypoints']

        # Example of how you might check "Receiver leaves after runner passes marker"
        # In practice, you'd see if the "Runner" keypoints have crossed some x-coord.

        # Pseudocode example:
        # runner_hip = get_keypoint(kpts, runner_hip_idx)
        # marker_x = 0.5  # Some reference
        # if runner_hip and runner_hip[0] > marker_x:
        #     # Means runner crossed the marker
        #     # Then check receiver's orientation or hand extension
        #     partial_scoring['departures_after_marker'] = 1
        #     partial_eval_frames[1].append(frame)

        logger.debug(f"Frame {frame}: Checking if runner passed marker, etc... (placeholder logic)")

    logger.debug(f"Final scoring for Phase A: {partial_scoring}")
    return partial_scoring, partial_eval_frames

def evaluate_phase_b(phase_b_frames):
    """
    Phase B (Acceleration):
    - Criterion 2: "Stick receiver reaches maximum speed"
    """
    partial_scoring = {
        'receiver_max_speed': 0
    }
    partial_eval_frames = {
        2: []
    }

    logger.debug(f"PHASE=B: Checking Criterion 2 over {len(phase_b_frames)} frames.")

    for idx, data in enumerate(phase_b_frames):
        frame = data.get('frame', idx)
        kpts = data['keypoints']

        # Example: measure velocity changes for the "receiver" by comparing consecutive frames' positions
        # receiver_hip = get_keypoint(kpts, 12) # just an example
        # measure speed or frame differences
        logger.debug(f"Frame {frame}: Checking receiver speed... (placeholder logic)")

    logger.debug(f"Final scoring for Phase B: {partial_scoring}")
    return partial_scoring, partial_eval_frames

def evaluate_phase_c(phase_c_frames):
    """
    Phase C (Exchange):
    - Criterion 3: "Change of baton takes place at speed after an agreed signal"
    - Criterion 4: "After receiving baton, baton changes hands. The runner remains in lane"
    - Criterion 5: "Change takes place within the zone"
    """
    partial_scoring = {
        'exchange_at_speed_signal': 0,
        'baton_hand_change_lane': 0,
        'exchange_in_zone': 0
    }
    partial_eval_frames = {
        3: [],
        4: [],
        5: []
    }

    logger.debug(f"PHASE=C: Checking Criteria 3,4,5 over {len(phase_c_frames)} frames.")

    for idx, data in enumerate(phase_c_frames):
        frame = data.get('frame', idx)
        kpts = data['keypoints']

        # Criterion 3: speed + signal
        # Possibly check the difference in the runner's position between consecutive frames
        # or a "signal" if you can detect an arm wave or call? (placeholder)
        logger.debug(f"Frame {frame}: Checking baton exchange speed & signal... (placeholder)")

        # if condition:
        #     partial_scoring['exchange_at_speed_signal'] = 1
        #     partial_eval_frames[3].append(frame)

        # Criterion 4: "After receiving the baton, changes hands, runner in lane"
        # placeholder for checking lane boundaries, baton hand transitions
        logger.debug(f"Frame {frame}: Checking baton hand switch, lane compliance... (placeholder)")

        # if condition:
        #     partial_scoring['baton_hand_change_lane'] = 1
        #     partial_eval_frames[4].append(frame)

        # Criterion 5: "Change in zone"
        # Possibly check the position relative to exchange zone coords (placeholder)
        logger.debug(f"Frame {frame}: Checking if exchange took place within zone... (placeholder)")

        # if condition:
        #     partial_scoring['exchange_in_zone'] = 1
        #     partial_eval_frames[5].append(frame)

    logger.debug(f"Final scoring for Phase C: {partial_scoring}")
    return partial_scoring, partial_eval_frames

# ------------- main evaluation --------------------

def evaluate_relay_receiver(player_coords):
    """
    Main function to evaluate Relay Receiver performance based on player coordinates.
    This parallels the shot_put evaluation structure.
    """
    logger.info("Starting Relay Receiver evaluation.")

    # 1) detect phase transitions
    a_end_index, b_end_index = detect_phase_transitions(player_coords)
    logger.debug(f"a_end_index={a_end_index}, b_end_index={b_end_index}")

    # 2) segment frames by phase
    phase_a_frames, phase_b_frames, phase_c_frames = segment_video_into_phases(
        player_coords, a_end_index, b_end_index
    )

    # 3) evaluate each phase
    a_scoring, a_eval_frames = evaluate_phase_a(phase_a_frames)
    b_scoring, b_eval_frames = evaluate_phase_b(phase_b_frames)
    c_scoring, c_eval_frames = evaluate_phase_c(phase_c_frames)

    # 4) Merge results
    # You have 5 total criteria:
    #   1 => a_scoring['departures_after_marker']
    #   2 => b_scoring['receiver_max_speed']
    #   3 => c_scoring['exchange_at_speed_signal']
    #   4 => c_scoring['baton_hand_change_lane']
    #   5 => c_scoring['exchange_in_zone']

    scoring = {
        'departures_after_marker': a_scoring.get('departures_after_marker', 0),
        'receiver_max_speed': b_scoring.get('receiver_max_speed', 0),
        'exchange_at_speed_signal': c_scoring.get('exchange_at_speed_signal', 0),
        'baton_hand_change_lane': c_scoring.get('baton_hand_change_lane', 0),
        'exchange_in_zone': c_scoring.get('exchange_in_zone', 0)
    }

    logger.debug(f"Merged scoring: {scoring}")

    eval_frames = {
        1: a_eval_frames.get(1, []),
        2: b_eval_frames.get(2, []),
        3: c_eval_frames.get(3, []),
        4: c_eval_frames.get(4, []),
        5: c_eval_frames.get(5, [])
    }

    logger.info("Relay Receiver evaluation completed.")
    return scoring, eval_frames
