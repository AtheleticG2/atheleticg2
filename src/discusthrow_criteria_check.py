import math
import logging

# ------------- logging config --------------------

#configure logging at the top of your module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  #capture all levels of logs

#create handlers
file_handler = logging.FileHandler('discus_throw_debug.log')
file_handler.setLevel(logging.DEBUG)

#console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  

#create formatters and add them to handlers
formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

#add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ------------- helper geometry functions --------------------
def get_keypoint(kpts, idx):

    if idx < len(kpts):
        return kpts[idx]
    return None

def distance_2d(p1, p2):
    if p1 is None or p2 is None:
        return None
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def compute_angle_3pts(a, b, c):

    if a is None or b is None or c is None:
        return None
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cx, cy = c[0], c[1]
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
    if not player_coords:
        logger.debug("player_coords is empty.")
        return 0, 0

    total = len(player_coords)
    one_third = total // 3
    two_thirds = (2 * total) // 3

    #use indices directly
    swing_end_index = one_third
    turn_end_index = two_thirds

    logger.debug(f"swing_end_index={swing_end_index}, turn_end_index={turn_end_index}")

    return swing_end_index, turn_end_index

def segment_video_into_phases(player_coords, swing_end_index, turn_end_index):

    swing_phase_frames = player_coords[:swing_end_index]
    turn_phase_frames = player_coords[swing_end_index:turn_end_index]
    throw_phase_frames = player_coords[turn_end_index:]

    logger.debug(f"Segments: swing_frames={len(swing_phase_frames)}, turn_frames={len(turn_phase_frames)}, throw_frames={len(throw_phase_frames)}")

    return swing_phase_frames, turn_phase_frames, throw_phase_frames

# ------------- criterion checks by phase --------------------
def evaluate_swing_phase(swing_frames):

    partial_scoring = {
        'intro_swing_behind': 0
    }
    partial_eval_frames = {
        1: []
    }

    logger.debug(f"PHASE=Swing: Processing {len(swing_frames)} frames for Criterion 1.")

    for idx, data in enumerate(swing_frames):
        frame = data.get('frame', idx) 
        kpts = data['keypoints']

        #debug: which criterion
        logger.debug(f"PHASE=Swing Frame={frame}: Checking Criterion 1...")

        #extract keypoints
        right_shoulder = get_keypoint(kpts, 6)
        right_hip = get_keypoint(kpts, 12)
        right_wrist = get_keypoint(kpts, 10)

        #log extracted keypoints
        logger.debug(f"Frame {frame}: right_shoulder={right_shoulder}, right_hip={right_hip}, right_wrist={right_wrist}")

        #criterion 1: introductory swing behind
        if right_wrist and right_shoulder and right_hip:
            swing_angle_calc = compute_angle_3pts(right_wrist, right_shoulder, right_hip)
            if swing_angle_calc is None:
                logger.debug(f"Frame {frame}: Unable to compute swing_angle due to insufficient keypoints.")
            else:
                logger.debug(f"Frame {frame}: Computed swing_angle={swing_angle_calc:.2f} degrees")
                logger.debug(f"Frame {frame}: wrist_x={right_wrist[0]:.4f}, shoulder_x={right_shoulder[0]:.4f}")
                #make sure the arm is swing behind based on angle and position
                if swing_angle_calc > 160 and right_wrist[0] < right_shoulder[0]:
                    partial_scoring['intro_swing_behind'] = 1
                    partial_eval_frames[1].append(frame)
                    logger.debug(f"Frame {frame}: Criterion 1 passed.")
                else:
                    logger.debug(f"Frame {frame}: Criterion 1 failed: swing_angle={swing_angle_calc:.2f}, wrist_x={right_wrist[0]:.4f}, shoulder_x={right_shoulder[0]:.4f}")
        else:
            logger.debug(f"Frame {frame}: Missing keypoints required for Criterion 1 evaluation.")

    logger.debug(f"Final scoring for Swing phase: {partial_scoring}")
    return partial_scoring, partial_eval_frames


def evaluate_turn_phase(turn_frames):

    partial_scoring = {
        'jump_turn_initiated': 0,
        'jump_turn_center_circle': 0
    }
    partial_eval_frames = {
        2: [],
        3: []
    }

    logger.debug(f"PHASE=Turn: Processing {len(turn_frames)} frames for Criteria 2 & 3.")

    #circle center check
    circle_center_x = 0.42
    threshold_distance = 0.05

    for idx, data in enumerate(turn_frames):
        frame = data.get('frame', idx + len(turn_frames)) 
        kpts = data['keypoints']

        #debugwhich criteria
        logger.debug(f"PHASE=Turn Frame={frame}: Checking Criteria 2 & 3...")

        right_hip = get_keypoint(kpts, 12)
        right_knee = get_keypoint(kpts, 14)
        right_ankle = get_keypoint(kpts, 16)
        left_ankle = get_keypoint(kpts, 15)

        #compute angles debug only
        jump_angle = compute_angle_3pts(right_hip, right_knee, right_ankle)

        #debug lines
        logger.debug(f"Frame {frame}: jump_angle={jump_angle}")

        #criterion 2: jump turn initiated from ball of foot
        if right_ankle and right_knee and right_hip:
            jump_angle_calc = compute_angle_3pts(right_hip, right_knee, right_ankle)
            if jump_angle_calc is None:
                logger.debug(f"Frame {frame}: Missing keypoints for jump_angle calculation.")
            else:
                logger.debug(f"Frame {frame}: jump_angle={jump_angle_calc}")
                if jump_angle_calc > 80: #adjust when needed
                    partial_scoring['jump_turn_initiated'] = 1
                    partial_eval_frames[2].append(frame)
                else:
                    logger.debug(f"Frame {frame}: Criterion 2 failed: jump_angle={jump_angle_calc}")

        #criterion 3: jump turn near center of circle
        if right_ankle and left_ankle:
            mid_ankle_x = (right_ankle[0] + left_ankle[0]) / 2
            logger.debug(f"Frame {frame}: mid_ankle_x={mid_ankle_x}")
            if abs(mid_ankle_x - circle_center_x) < threshold_distance:
                logger.debug(f"Frame {frame}: Criterion 3 satisfied: mid_ankle_x={mid_ankle_x}")
                partial_scoring['jump_turn_center_circle'] = 1
                partial_eval_frames[3].append(frame)
            else:
                logger.debug(f"Frame {frame}: Criterion 3 failed: mid_ankle_x={mid_ankle_x}")

    return partial_scoring, partial_eval_frames

def evaluate_throw_phase(throw_frames):
    partial_scoring = {
        'throw_off_low_to_high': 0,
        'discus_release_via_wrist': 0
    }
    partial_eval_frames = {
        4: [],
        5: []
    }

    logger.debug(f"PHASE=Throw: Processing {len(throw_frames)} frames for Criteria 4 & 5.")

    for idx, data in enumerate(throw_frames):
        frame = data.get('frame', idx + len(throw_frames))
        kpts = data['keypoints']

        logger.debug(f"PHASE=Throw Frame={frame}: Checking Criteria 4 & 5...")

        right_knee = get_keypoint(kpts, 14)
        right_hip = get_keypoint(kpts, 12)
        left_knee = get_keypoint(kpts, 13)
        right_shoulder = get_keypoint(kpts, 6)
        right_wrist = get_keypoint(kpts, 10)

        #compute angles debug only
        throw_angle = compute_angle_3pts(right_knee, right_hip, left_knee)
        release_angle = compute_angle_3pts(right_shoulder, right_wrist, (right_wrist[0] + 1, right_wrist[1]))

        #debug log
        logger.debug(f"Frame {frame}: throw_angle={throw_angle}, release_angle={release_angle}")

        #criterion 4: throw-off from low to high
        if right_hip and right_knee and left_knee:
            throw_angle_calc = compute_angle_3pts(right_knee, right_hip, left_knee)
            if throw_angle_calc is None:
                logger.debug(f"Frame {frame}: Missing keypoints for throw_angle calculation.")
            else:
                logger.debug(f"Frame {frame}: throw_angle={throw_angle_calc}")
                if throw_angle_calc > 45:  # Adjust threshold as needed
                    partial_scoring['throw_off_low_to_high'] = 1
                    partial_eval_frames[4].append(frame)
                else:
                    logger.debug(f"Frame {frame}: Criterion 4 failed: throw_angle={throw_angle_calc}")

        #criterion 5: discus release via wrist
        if right_shoulder and right_wrist:
            #using a mock point (right_wrist[0] + 1, right_wrist[1]) for direction
            release_angle_calc = compute_angle_3pts(
                right_shoulder, 
                right_wrist, 
                (right_wrist[0] + 1, right_wrist[1])
            )
            if release_angle_calc is None:
                logger.debug(f"Frame {frame}: Missing keypoints for release_angle calculation.")
            else:
                logger.debug(f"Frame {frame}: release_angle={release_angle_calc}")
                if release_angle_calc > 30:  # Adjust threshold for wrist snap
                    partial_scoring['discus_release_via_wrist'] = 1
                    partial_eval_frames[5].append(frame)
                else:
                    logger.debug(f"Frame {frame}: Criterion 5 failed: release_angle={release_angle_calc}")

    return partial_scoring, partial_eval_frames

# ------------- main evaluation --------------------
def evaluate_discus_throw(player_coords):

    #1) detect phase transitions
    swing_end_index, turn_end_index = detect_phase_transitions(player_coords)
    logger.debug(f"swing_end_index={swing_end_index}, turn_end_index={turn_end_index}")

    #2) segment frames by phase
    swing_frames, turn_frames, throw_frames = segment_video_into_phases(
        player_coords, swing_end_index, turn_end_index
    )

    #3) evaluate each phase
    swing_scoring, swing_eval_frames = evaluate_swing_phase(swing_frames)
    turn_scoring, turn_eval_frames = evaluate_turn_phase(turn_frames)
    throw_scoring, throw_eval_frames = evaluate_throw_phase(throw_frames)

    #4) Merge results
    scoring = {
        'intro_swing_behind': swing_scoring.get('intro_swing_behind', 0),
        'jump_turn_initiated': turn_scoring.get('jump_turn_initiated', 0),
        'jump_turn_center_circle': turn_scoring.get('jump_turn_center_circle', 0),
        'throw_off_low_to_high': throw_scoring.get('throw_off_low_to_high', 0),
        'discus_release_via_wrist': throw_scoring.get('discus_release_via_wrist', 0)
    }

    #log merged scoring
    logger.debug(f"Merged scoring: {scoring}")


    eval_frames = {
        1: swing_eval_frames.get(1, []),
        2: turn_eval_frames.get(2, []),
        3: turn_eval_frames.get(3, []),
        4: throw_eval_frames.get(4, []),
        5: throw_eval_frames.get(5, [])
    }

    return scoring, eval_frames

