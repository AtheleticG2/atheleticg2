import numpy as np

# Function to calculate angles between three points


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle


# Function to retrieve a keypoint
def get_keypoint(keypoints, keypoint_index):
    try:
        return keypoints[keypoint_index].tolist()
    except (IndexError, AttributeError):
        return None

# Function to calculate the midpoint between two points


def get_midpoint(point1, point2):
    return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]


def javelin_draw_backward(arm_angle, threshold=90):
    """
    Check if the javelin arm is drawn back to an optimal angle.
    """
    if arm_angle > threshold:
        return True
    return False


def pelvis_rotation_inward(hip_angle, threshold=15):
    """
    Check if the pelvis rotates inward during the throwing motion.
    """
    if hip_angle < threshold:
        return True
    return False


def execute_impulse_step(ankle_angle, knee_angle, threshold=45):
    """
    Check if the impulse step is executed with appropriate angles.
    """
    if ankle_angle > threshold and knee_angle > threshold:
        return True
    return False


def execute_blocking_step(hip_angle, knee_angle, threshold=60):
    """
    Check if the blocking step is executed based on hip and knee angle criteria.
    """
    if hip_angle > threshold and knee_angle > threshold:
        return True
    return False


def initiate_throw_through_hips(pelvis_angle, torso_angle, threshold=30):
    """
    Ensure that the throw is initiated through the hips and torso after the blocking step.
    """
    if pelvis_angle > threshold and torso_angle < 60:
        return True
    return False


def last_5_strides(foot_position, stride_threshold=5):
    """
    Check if the player is in the last 5 strides before the impulse step.
    """
    stride_count = len(foot_position)
    if stride_count >= stride_threshold:
        # Check if foot position shows backward movement for last 5 steps
        recent_strides = foot_position[-stride_threshold:]
        if all(recent_strides[i][1] > recent_strides[i+1][1] for i in range(stride_threshold-1)):
            return True
    return False


def evaluate_javelin_throw(player_coords):
    scoring = {'pelvis_rotation_inward': 0, 'impulse_step_executed': 0,
               'blocking_step_executed': 0, 'throw_initiated_through_hips': 0, 'last_5_strides': 0}

    # Initialize lists for angle tracking
    pelvis_angles = []
    ankle_angles = []
    knee_angles = []
    foot_positions = []  # To track foot positions for last strides

    evaluation_frames = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

    for data in player_coords:
        frame = data['frame']
        keypoints = data['keypoints']
        left_hip = get_keypoint(keypoints, 11)
        right_hip = get_keypoint(keypoints, 12)
        left_knee = get_keypoint(keypoints, 13)
        left_ankle = get_keypoint(keypoints, 15)
        right_knee = get_keypoint(keypoints, 14)
        right_ankle = get_keypoint(keypoints, 16)

        if not (left_hip and right_hip and left_knee and left_ankle and right_knee and right_ankle):
            continue

        # Track foot positions for stride calculation
        # Example index, change based on actual keypoints
        left_foot = get_keypoint(keypoints, 17)
        # Example index, change based on actual keypoints
        right_foot = get_keypoint(keypoints, 18)

        if left_foot and right_foot:
            foot_positions.append([frame, left_foot, right_foot])

        # Calculate relevant angles
        pelvis_angle = calculate_angle(left_hip, left_knee, left_ankle)
        ankle_angle = calculate_angle(left_ankle, left_knee, left_hip)
        knee_angle = calculate_angle(left_knee, left_hip, right_hip)

        pelvis_angles.append(pelvis_angle)
        ankle_angles.append(ankle_angle)
        knee_angles.append(knee_angle)

        # Criterion 1: Pelvis rotates inward
        if pelvis_rotation_inward(pelvis_angle):
            scoring['pelvis_rotation_inward'] = 1
            evaluation_frames[1].append(frame)

        # Criterion 2: Impulse step execution
        if execute_impulse_step(ankle_angle, knee_angle):
            scoring['impulse_step_executed'] = 1
            evaluation_frames[2].append(frame)

        # Criterion 3: Blocking step execution
        if execute_blocking_step(pelvis_angle, knee_angle):
            scoring['blocking_step_executed'] = 1
            evaluation_frames[3].append(frame)

        # Criterion 4: Throw initiated through hips
        if initiate_throw_through_hips(pelvis_angle, knee_angle):
            scoring['throw_initiated_through_hips'] = 1
            evaluation_frames[4].append(frame)

        # Criterion 5: Last 5 strides check
        if last_5_strides(foot_positions):
            scoring['last_5_strides'] = 1
            evaluation_frames[5].append(frame)

    return scoring, evaluation_frames
