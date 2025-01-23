import numpy as np

# Function to calculate angles between three points (same as before)


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

# Function to retrieve a keypoint (same as before)


def get_keypoint(keypoints, keypoint_index):
    try:
        return keypoints[keypoint_index].tolist()
    except (IndexError, AttributeError):
        return None

# Function to calculate the midpoint between two points


def get_midpoint(point1, point2):
    return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]

# Function to calculate the distance between two points


def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to check if the javelin is drawn backward during the last 5 strides


def javelin_drawn_backward(shoulder_positions, wrist_positions, last_n_frames=5):
    if len(shoulder_positions) < last_n_frames or len(wrist_positions) < last_n_frames:
        return False

    # Check if the wrist moves backward relative to the shoulder over the last N frames
    wrist_movement = wrist_positions[-1][0] - \
        wrist_positions[-last_n_frames][0]
    shoulder_movement = shoulder_positions[-1][0] - \
        shoulder_positions[-last_n_frames][0]

    return wrist_movement < shoulder_movement

# Function to check pelvis rotation and javelin fully drawn back


def pelvis_rotation_and_javelin_drawn(hip_positions, shoulder_positions, wrist_positions):
    if len(hip_positions) < 2 or len(shoulder_positions) < 2 or len(wrist_positions) < 2:
        return False

    # Check if the pelvis rotates inward (hips move closer to the midline)
    hip_movement = hip_positions[-1][0] - hip_positions[-2][0]
    shoulder_movement = shoulder_positions[-1][0] - shoulder_positions[-2][0]

    # Check if the wrist is fully drawn back (behind the shoulder)
    wrist_behind_shoulder = wrist_positions[-1][0] < shoulder_positions[-1][0]

    return hip_movement < shoulder_movement and wrist_behind_shoulder

# Function to check the execution of the impulse step


def impulse_step_executed(ankle_positions, knee_positions, hip_positions):
    if len(ankle_positions) < 2 or len(knee_positions) < 2 or len(hip_positions) < 2:
        return False

    # Check if the ankle moves forward relative to the knee and hip
    ankle_movement = ankle_positions[-1][0] - ankle_positions[-2][0]
    knee_movement = knee_positions[-1][0] - knee_positions[-2][0]
    hip_movement = hip_positions[-1][0] - hip_positions[-2][0]

    return ankle_movement > knee_movement and ankle_movement > hip_movement

# Function to check the execution of the blocking step


def blocking_step_executed(foot_positions, hip_positions):
    if len(foot_positions) < 2 or len(hip_positions) < 2:
        return False

    # Check if the foot stops moving forward while the hips continue to rotate
    foot_movement = foot_positions[-1][0] - foot_positions[-2][0]
    hip_movement = hip_positions[-1][0] - hip_positions[-2][0]

    return abs(foot_movement) < 0.1 and abs(hip_movement) > 0.1

# Function to check the initiation of the throw through hips and torso


def throw_initiated(hip_positions, shoulder_positions, wrist_positions):
    if len(hip_positions) < 2 or len(shoulder_positions) < 2 or len(wrist_positions) < 2:
        return False

    # Check if the hips and torso rotate forward while the wrist moves forward
    hip_movement = hip_positions[-1][0] - hip_positions[-2][0]
    shoulder_movement = shoulder_positions[-1][0] - shoulder_positions[-2][0]
    wrist_movement = wrist_positions[-1][0] - wrist_positions[-2][0]

    return hip_movement > 0 and shoulder_movement > 0 and wrist_movement > 0

# Main evaluation function for javelin throw


def evaluate_javelin_throw(player_coords):
    scoring = {
        'javelin_drawn_backward': 0,
        'pelvis_rotation_and_javelin_drawn': 0,
        'impulse_step_executed': 0,
        'blocking_step_executed': 0,
        'throw_initiated': 0
    }

    evaluation_frames = {1: [], 2: [], 3: [], 4: [], 5: []}

    # Initialize lists to track positions over time
    shoulder_positions = []
    wrist_positions = []
    hip_positions = []
    knee_positions = []
    ankle_positions = []
    foot_positions = []

    for data in player_coords:
        frame = data['frame']
        keypoints = data['keypoints']

        # Get relevant keypoints
        left_shoulder = get_keypoint(keypoints, 5)
        right_shoulder = get_keypoint(keypoints, 6)
        left_wrist = get_keypoint(keypoints, 9)
        right_wrist = get_keypoint(keypoints, 10)
        left_hip = get_keypoint(keypoints, 11)
        right_hip = get_keypoint(keypoints, 12)
        left_knee = get_keypoint(keypoints, 13)
        right_knee = get_keypoint(keypoints, 14)
        left_ankle = get_keypoint(keypoints, 15)
        right_ankle = get_keypoint(keypoints, 16)
        left_foot = get_keypoint(keypoints, 17)
        right_foot = get_keypoint(keypoints, 18)

        # Use midpoints for shoulders, hips, knees, ankles, and feet
        mid_shoulder = get_midpoint(
            left_shoulder, right_shoulder) if left_shoulder and right_shoulder else None
        mid_wrist = get_midpoint(
            left_wrist, right_wrist) if left_wrist and right_wrist else None
        mid_hip = get_midpoint(
            left_hip, right_hip) if left_hip and right_hip else None
        mid_knee = get_midpoint(
            left_knee, right_knee) if left_knee and right_knee else None
        mid_ankle = get_midpoint(
            left_ankle, right_ankle) if left_ankle and right_ankle else None
        mid_foot = get_midpoint(
            left_foot, right_foot) if left_foot and right_foot else None

        # Append positions to lists if all required keypoints are available
        if mid_shoulder and mid_wrist and mid_hip and mid_knee and mid_ankle and mid_foot:
            shoulder_positions.append(mid_shoulder)
            wrist_positions.append(mid_wrist)
            hip_positions.append(mid_hip)
            knee_positions.append(mid_knee)
            ankle_positions.append(mid_ankle)
            foot_positions.append(mid_foot)

            # Criterion 1: Javelin drawn backward during the last 5 strides
            if javelin_drawn_backward(shoulder_positions, wrist_positions):
                scoring['javelin_drawn_backward'] = 1
                evaluation_frames[1].append(frame)

            # Criterion 2: Pelvis rotates inward, and javelin is fully drawn back
            if pelvis_rotation_and_javelin_drawn(hip_positions, shoulder_positions, wrist_positions):
                scoring['pelvis_rotation_and_javelin_drawn'] = 1
                evaluation_frames[2].append(frame)

            # Criterion 3: Execution of the impulse step
            if impulse_step_executed(ankle_positions, knee_positions, hip_positions):
                scoring['impulse_step_executed'] = 1
                evaluation_frames[3].append(frame)

            # Criterion 4: Execution of the blocking step
            if blocking_step_executed(foot_positions, hip_positions):
                scoring['blocking_step_executed'] = 1
                evaluation_frames[4].append(frame)

            # Criterion 5: Throw initiated through hips and torso
            if throw_initiated(hip_positions, shoulder_positions, wrist_positions):
                scoring['throw_initiated'] = 1
                evaluation_frames[5].append(frame)

    return scoring, evaluation_frames
