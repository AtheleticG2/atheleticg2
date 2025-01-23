import numpy as np

# Function to calculate angles between three points


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle


def get_keypoint(keypoints, keypoint_index):
    try:
        return keypoints[keypoint_index].tolist()
    except (IndexError, AttributeError):
        return None


def javelin_drawn_backward(shoulder_positions, wrist_positions, side, last_n_frames=5):
    if len(shoulder_positions) < last_n_frames or len(wrist_positions) < last_n_frames:
        print(
            f"javelin_drawn_backward ({side}): Insufficient data. Shoulder: {len(shoulder_positions)}, Wrist: {len(wrist_positions)}")
        return False

    wrist_movement = wrist_positions[-1][0] - \
        wrist_positions[-last_n_frames][0]
    shoulder_movement = shoulder_positions[-1][0] - \
        shoulder_positions[-last_n_frames][0]

    if wrist_movement >= shoulder_movement:
        print(
            f"javelin_drawn_backward ({side}): Wrist movement ({wrist_movement}) >= shoulder ({shoulder_movement})")
        return False
    return True


def pelvis_rotation_and_javelin_drawn(hip_positions, shoulder_positions, wrist_positions, side):
    if len(hip_positions) < 2 or len(shoulder_positions) < 2 or len(wrist_positions) < 2:
        print(
            f"pelvis_rotation ({side}): Insufficient data. Hip: {len(hip_positions)}, Shoulder: {len(shoulder_positions)}, Wrist: {len(wrist_positions)}")
        return False

    hip_movement = hip_positions[-1][0] - hip_positions[-2][0]
    shoulder_movement = shoulder_positions[-1][0] - shoulder_positions[-2][0]
    wrist_behind = wrist_positions[-1][0] < shoulder_positions[-1][0]

    if not (hip_movement < shoulder_movement and wrist_behind):
        print(
            f"pelvis_rotation ({side}): Hip move ({hip_movement}) >= shoulder ({shoulder_movement}) or wrist not behind")
        return False
    return True


def impulse_step_executed(ankle_positions, knee_positions, hip_positions, side):
    if len(ankle_positions) < 2 or len(knee_positions) < 2 or len(hip_positions) < 2:
        print(
            f"impulse_step ({side}): Insufficient data. Ankle: {len(ankle_positions)}, Knee: {len(knee_positions)}, Hip: {len(hip_positions)}")
        return False

    ankle_move = ankle_positions[-1][0] - ankle_positions[-2][0]
    knee_move = knee_positions[-1][0] - knee_positions[-2][0]
    hip_move = hip_positions[-1][0] - hip_positions[-2][0]

    if not (ankle_move > knee_move and ankle_move > hip_move):
        print(
            f"impulse_step ({side}): Ankle ({ankle_move}) <= knee ({knee_move}) or hip ({hip_move})")
        return False
    return True


def blocking_step_executed(ankle_positions, hip_positions, side):
    if len(ankle_positions) < 2 or len(hip_positions) < 2:
        print(
            f"blocking_step ({side}): Insufficient data. Ankle: {len(ankle_positions)}, Hip: {len(hip_positions)}")
        return False

    ankle_move = abs(ankle_positions[-1][0] - ankle_positions[-2][0])
    hip_move = abs(hip_positions[-1][0] - hip_positions[-2][0])

    if not (ankle_move < 0.1 and hip_move > 0.1):
        print(
            f"blocking_step ({side}): Ankle move ({ankle_move}) >= 0.1 or hip move ({hip_move}) <= 0.1")
        return False
    return True


def throw_initiated(hip_positions, shoulder_positions, wrist_positions, side):
    if len(hip_positions) < 2 or len(shoulder_positions) < 2 or len(wrist_positions) < 2:
        print(
            f"throw_initiated ({side}): Insufficient data. Hip: {len(hip_positions)}, Shoulder: {len(shoulder_positions)}, Wrist: {len(wrist_positions)}")
        return False

    hip_move = hip_positions[-1][0] - hip_positions[-2][0]
    shoulder_move = shoulder_positions[-1][0] - shoulder_positions[-2][0]
    wrist_move = wrist_positions[-1][0] - wrist_positions[-2][0]

    if not (hip_move > 0 and shoulder_move > 0 and wrist_move > 0):
        print(
            f"throw_initiated ({side}): Hip ({hip_move}), shoulder ({shoulder_move}), or wrist ({wrist_move}) <= 0")
        return False
    return True


def evaluate_javelin_throw(player_coords):
    scoring = {
        'javelin_drawn_backward': 0,
        'pelvis_rotation_and_javelin_drawn': 0,
        'impulse_step_executed': 0,
        'blocking_step_executed': 0,
        'throw_initiated': 0
    }

    evaluation_frames = {1: [], 2: [], 3: [], 4: [], 5: []}

    # Initialize trackers for COCO keypoints (indices 0-16 only)
    left_shoulder, right_shoulder = [], []
    left_wrist, right_wrist = [], []
    left_hip, right_hip = [], []
    left_knee, right_knee = [], []
    left_ankle, right_ankle = [], []

    for data in player_coords:
        frame = data['frame']
        keypoints = data['keypoints']

        # Get COCO-compliant keypoints
        ls = get_keypoint(keypoints, 5)   # left_shoulder
        rs = get_keypoint(keypoints, 6)   # right_shoulder
        lw = get_keypoint(keypoints, 9)   # left_wrist
        rw = get_keypoint(keypoints, 10)  # right_wrist
        lh = get_keypoint(keypoints, 11)  # left_hip
        rh = get_keypoint(keypoints, 12)  # right_hip
        lk = get_keypoint(keypoints, 13)  # left_knee
        rk = get_keypoint(keypoints, 14)  # right_knee
        la = get_keypoint(keypoints, 15)  # left_ankle
        ra = get_keypoint(keypoints, 16)  # right_ankle

        # Update trackers
        if ls:
            left_shoulder.append(ls)
        if rs:
            right_shoulder.append(rs)
        if lw:
            left_wrist.append(lw)
        if rw:
            right_wrist.append(rw)
        if lh:
            left_hip.append(lh)
        if rh:
            right_hip.append(rh)
        if lk:
            left_knee.append(lk)
        if rk:
            right_knee.append(rk)
        if la:
            left_ankle.append(la)
        if ra:
            right_ankle.append(ra)

        # Criterion 1: Javelin drawn backward
        crit1_left = javelin_drawn_backward(left_shoulder, left_wrist, 'left')
        crit1_right = javelin_drawn_backward(
            right_shoulder, right_wrist, 'right')
        if crit1_left or crit1_right:
            scoring['javelin_drawn_backward'] = 1
            evaluation_frames[1].append(frame)

        # Criterion 2: Pelvis rotation + javelin drawn back
        crit2_left = pelvis_rotation_and_javelin_drawn(
            left_hip, left_shoulder, left_wrist, 'left')
        crit2_right = pelvis_rotation_and_javelin_drawn(
            right_hip, right_shoulder, right_wrist, 'right')
        if crit2_left or crit2_right:
            scoring['pelvis_rotation_and_javelin_drawn'] = 1
            evaluation_frames[2].append(frame)

        # Criterion 3: Impulse step executed
        crit3_left = impulse_step_executed(
            left_ankle, left_knee, left_hip, 'left')
        crit3_right = impulse_step_executed(
            right_ankle, right_knee, right_hip, 'right')
        if crit3_left or crit3_right:
            scoring['impulse_step_executed'] = 1
            evaluation_frames[3].append(frame)

        # Criterion 4: Blocking step executed
        crit4_left = blocking_step_executed(left_ankle, left_hip, 'left')
        crit4_right = blocking_step_executed(right_ankle, right_hip, 'right')
        if crit4_left or crit4_right:
            scoring['blocking_step_executed'] = 1
            evaluation_frames[4].append(frame)

        # Criterion 5: Throw initiated
        crit5_left = throw_initiated(
            left_hip, left_shoulder, left_wrist, 'left')
        crit5_right = throw_initiated(
            right_hip, right_shoulder, right_wrist, 'right')
        if crit5_left or crit5_right:
            scoring['throw_initiated'] = 1
            evaluation_frames[5].append(frame)

    return scoring, evaluation_frames
