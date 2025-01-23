import math

def get_keypoint(kpts, idx):
    if idx < len(kpts):
        return kpts[idx]
    return None

def distance_2d(p1, p2):
    """Euclidean distance in 2D, or None if p1/p2 are missing."""
    if p1 is None or p2 is None:
        return None
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def compute_angle_3pts(a, b, c):
    """
    Computes the angle (in degrees) at point b formed by points a->b->c.
    Returns None if any point is missing or if the vectors are too short.
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
    # Clamp cos_angle to [-1,1] to avoid floating-point errors
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    angle_deg = math.degrees(math.acos(cos_angle))
    return angle_deg

def compute_reference_measurement(player_coords):
    """
    Computes the average hip width across all frames to use as a normalization factor.
    
    :param player_coords: List of dicts, each with 'frame' and 'keypoints'
    :return: Average hip width
    """
    hip_widths = []
    for data in player_coords:
        kpts = data['keypoints']
        right_hip = get_keypoint(kpts, 12)
        left_hip = get_keypoint(kpts, 11)
        if right_hip and left_hip:
            hip_width = distance_2d(right_hip, left_hip)
            if hip_width:
                hip_widths.append(hip_width)
    if hip_widths:
        average_hip_width = sum(hip_widths) / len(hip_widths)
        return average_hip_width
    else:
        return None

def normalize_keypoints(player_coords, reference_measurement):
    """
    Normalizes keypoint coordinates based on the reference measurement (e.g., hip width).
    
    :param player_coords: List of dicts, each with 'frame' and 'keypoints'
    :param reference_measurement: Float value for normalization
    :return: List of dicts with normalized 'keypoints'
    """
    normalized_coords = []
    for data in player_coords:
        frame = data['frame']
        kpts = data['keypoints']
        normalized_kpts = []
        for kp in kpts:
            if kp:
                normalized_kpts.append((kp[0] / reference_measurement, kp[1] / reference_measurement))
            else:
                normalized_kpts.append(None)
        normalized_coords.append({'frame': frame, 'keypoints': normalized_kpts})
    return normalized_coords

def evaluate_discus_throw_normalized(
    player_coords,
    # Parametric thresholds
    swing_angle_thresh=150,         # for "intro_swing_behind"
    jump_angle_thresh=65,           # for "jump_turn_initiated"
    circle_center=(0.42, 0.5),      # (x,y) circle center in normalized coords
    circle_dist_thresh=0.1,        # in 'hip-width' units
    throw_angle_range=(110, 160),   # for "throw_off_low_to_high" (min, max)
    release_angle_thresh=135        # for "discus_leaves_index_finger"
):
    """
    Evaluates the discus throw using normalized keypoints.
    
    :param player_coords: List of dicts with 'frame' and 'keypoints' (normalized)
    :return: (scoring, eval_frames)
    """
    scoring = {
        'intro_swing_behind': 0,
        'jump_turn_initiated': 0,
        'jump_turn_center_circle': 0,
        'throw_off_low_to_high': 0,
        'discus_leaves_index_finger': 0
    }
    eval_frames = {1: [], 2: [], 3: [], 4: [], 5: []}

    # Initialize tracking variables
    max_swing_angle = 0.0
    max_jump_angle = 0.0
    max_throw_angle = 0.0
    max_release_angle = 0.0
    min_center_dist_ratio = float('inf')

    # Frame tracking
    best_swing_frame = None
    best_jump_frame = None
    best_throw_frame = None
    best_release_frame = None
    best_center_frame = None

    for data in player_coords:
        frame = data['frame']
        kpts = data['keypoints']

        # Extract keypoints
        right_shoulder = get_keypoint(kpts, 6)
        right_hip = get_keypoint(kpts, 12)
        right_knee = get_keypoint(kpts, 14)
        right_ankle = get_keypoint(kpts, 16)
        left_hip = get_keypoint(kpts, 11)
        left_knee = get_keypoint(kpts, 13)
        left_ankle = get_keypoint(kpts, 15)
        right_wrist = get_keypoint(kpts, 10)

        # Compute angles
        swing_angle = compute_angle_3pts(right_wrist, right_shoulder, right_hip)
        jump_angle = compute_angle_3pts(right_hip, right_knee, right_ankle)
        throw_angle = compute_angle_3pts(right_knee, right_hip, left_knee)
        release_angle = compute_angle_3pts(right_shoulder, right_wrist, right_ankle)

        # Debug information
        print(f"[DEBUG Frame={frame}] swing_angle={swing_angle}, jump_angle={jump_angle}, "
              f"throw_angle={throw_angle}, release_angle={release_angle}")

        # Update max angles
        if swing_angle is not None and swing_angle > max_swing_angle:
            max_swing_angle = swing_angle
            best_swing_frame = frame

        if jump_angle is not None and jump_angle > max_jump_angle:
            max_jump_angle = jump_angle
            best_jump_frame = frame

        if throw_angle is not None and throw_angle > max_throw_angle:
            max_throw_angle = throw_angle
            best_throw_frame = frame

        if release_angle is not None and release_angle > max_release_angle:
            max_release_angle = release_angle
            best_release_frame = frame

        # Criterion 3: Jump turn near center of circle (normalized)
        if right_ankle and left_ankle:
            mid_ankle_x = (right_ankle[0] + left_ankle[0]) / 2.0
            mid_ankle_y = (right_ankle[1] + left_ankle[1]) / 2.0
            dist_center = math.hypot(mid_ankle_x - circle_center[0], 
                                     mid_ankle_y - circle_center[1])
            ratio = dist_center  # Already normalized by hip width during normalization

            print(f"[DEBUG Frame={frame}] mid_ankle_x={mid_ankle_x}, mid_ankle_y={mid_ankle_y}, "
                  f"dist_center={dist_center}")

            if ratio < min_center_dist_ratio:
                min_center_dist_ratio = ratio
                best_center_frame = frame

    # Final scoring based on max/min values
    scoring['intro_swing_behind'] = 1 if max_swing_angle > swing_angle_thresh else 0
    scoring['jump_turn_initiated'] = 1 if max_jump_angle > jump_angle_thresh else 0
    scoring['jump_turn_center_circle'] = 1 if min_center_dist_ratio < circle_dist_thresh else 0
    scoring['throw_off_low_to_high'] = 1 if (throw_angle_range[0] <= max_throw_angle <= throw_angle_range[1]) else 0
    scoring['discus_leaves_index_finger'] = 1 if max_release_angle > release_angle_thresh else 0

    # Assign frames where criteria were met
    eval_frames = {
        1: [best_swing_frame] if scoring['intro_swing_behind'] else [],
        2: [best_jump_frame] if scoring['jump_turn_initiated'] else [],
        3: [best_center_frame] if scoring['jump_turn_center_circle'] else [],
        4: [best_throw_frame] if scoring['throw_off_low_to_high'] else [],
        5: [best_release_frame] if scoring['discus_leaves_index_finger'] else []
    }

    # Debug final summary
    print("=== Final Scoring ===")
    for key, val in scoring.items():
        print(f"{key}: {val}")
    print("Min center diff found:", min_center_dist_ratio)
    print("Max angles - swing:", max_swing_angle,
          "jump:", max_jump_angle,
          "throw:", max_throw_angle,
          "release:", max_release_angle)

    return scoring, eval_frames
