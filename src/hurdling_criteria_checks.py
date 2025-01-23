import numpy as np
from scipy.signal import find_peaks

# Function to denormalize keypoints


def denormalize_keypoint(keypoint, image_width, image_height):
    """
    Convert normalized keypoint coordinates to pixel values.

    Args:
        keypoint (tuple): Normalized keypoint (x, y, confidence).
        image_width (int): Width of the image/frame.
        image_height (int): Height of the image/frame.

    Returns:
        tuple: Pixel coordinates (x, y).
    """
    x, y, _ = keypoint  # Ignore confidence score
    return int(x * image_width), int(y * image_height)

# Function to retrieve a keypoint


def get_keypoint(keypoints, keypoint_index):
    """
    Retrieve a specific keypoint from the keypoints list.

    Args:
        keypoints (list): List of keypoints from the pose estimation model.
        keypoint_index (int): Index of the keypoint to retrieve.

    Returns:
        tuple: Keypoint coordinates (x, y) or None if not found.
    """
    try:
        return keypoints[keypoint_index].tolist()
    except (IndexError, AttributeError):
        return None

# Function to calculate the midpoint between two points


def get_midpoint(point1, point2):
    """
    Calculate the midpoint between two points.

    Args:
        point1 (tuple): First point (x, y).
        point2 (tuple): Second point (x, y).

    Returns:
        tuple: Midpoint coordinates (x, y).
    """
    if point1 is None or point2 is None:
        return None
    return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]

# Function to calculate angles between three points


def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.

    Args:
        a (tuple): First point (x, y).
        b (tuple): Second point (x, y).
        c (tuple): Third point (x, y).

    Returns:
        float: Angle in degrees.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

# Function to calculate the distance between two points


def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        point1 (tuple): First point (x, y).
        point2 (tuple): Second point (x, y).

    Returns:
        float: Distance between the points.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to detect strides


def detect_strides(ankle_positions, stride_threshold=50):
    """
    Detect strides based on ankle positions.

    Args:
        ankle_positions (list): List of ankle positions over time.
        stride_threshold (float): Minimum distance between ankles to consider it a stride.

    Returns:
        list: Indices of frames where strides occur.
    """
    if len(ankle_positions) < 2:
        return []

    # Calculate the distance between ankles in consecutive frames
    distances = [calculate_distance(
        ankle_positions[i], ankle_positions[i - 1]) for i in range(1, len(ankle_positions))]

    # Detect peaks in the distances to identify strides
    peaks, _ = find_peaks(distances, height=stride_threshold)
    return peaks

# Function to check lead leg and torso alignment


def check_lead_leg_and_torso(lead_knee, lead_ankle, torso_angle, arm_position, hurdle_height):
    """
    Check if the lead leg is almost straight and passes just above the hurdle,
    and the torso and opposite arm move toward the lead leg.

    Args:
        lead_knee (list): Position of the lead knee.
        lead_ankle (list): Position of the lead ankle.
        torso_angle (float): Angle of the torso.
        arm_position (list): Position of the opposite arm.
        hurdle_height (float): Predefined height of the hurdle.

    Returns:
        bool: True if the criteria are met, False otherwise.
    """
    # Check if the lead leg is almost straight
    lead_leg_angle = calculate_angle(lead_knee, lead_ankle, [
                                     lead_ankle[0], lead_ankle[1] - 10])  # Vertical reference
    is_straight = 160 <= lead_leg_angle <= 180

    # Check if the lead leg passes just above the hurdle
    leg_height = lead_ankle[1] - lead_knee[1]
    is_above_hurdle = leg_height > hurdle_height

    # Check if the torso and opposite arm move toward the lead leg
    is_torso_aligned = torso_angle < 20  # Example threshold
    is_arm_moving = arm_position[0] < lead_knee[0]  # Example threshold

    return is_straight and is_above_hurdle and is_torso_aligned and is_arm_moving

# Function to check second contact after hurdle


def check_second_contact(ankle_positions, knee_positions, stride_threshold=100, knee_height_threshold=150):
    """
    Check if the second contact after the hurdle involves a large stride with a visibly high knee.

    Args:
        ankle_positions (list): List of ankle positions over time.
        knee_positions (list): List of knee positions over time.
        stride_threshold (float): Minimum stride length to consider it large.
        knee_height_threshold (float): Minimum knee height to consider it visibly high.

    Returns:
        bool: True if the criteria are met, False otherwise.
    """
    if len(ankle_positions) < 2 or len(knee_positions) < 2:
        return False

    # Calculate stride length
    stride_length = calculate_distance(
        ankle_positions[-1], ankle_positions[-2])

    # Check knee height
    knee_height = knee_positions[-1][1]

    return stride_length > stride_threshold and knee_height > knee_height_threshold

# Main evaluation function for hurdling


def evaluate_hurdling(player_coords, hurdle_height=100):
    """
    Evaluate hurdling technique based on keypoints.

    Args:
        player_coords (list): List of keypoints for each frame.
        hurdle_height (float): Predefined height of the hurdle.

    Returns:
        dict: Scoring for each criterion.
        dict: Frames where each criterion is met.
    """
    scoring = {
        'approach_8_strides': 0,
        '4_contacts_between_hurdles': 0,
        'lead_leg_and_torso_alignment': 0,
        'second_contact_after_hurdle': 0
    }

    evaluation_frames = {1: [], 2: [], 3: [], 4: []}

    # Initialize lists to track positions and angles over time
    ankle_positions = []
    knee_positions = []
    lead_knee_positions = []
    torso_angles = []
    arm_positions = []

    for data in player_coords:
        frame = data['frame']
        keypoints = data['keypoints']

        # Get relevant keypoints
        left_knee = get_keypoint(keypoints, 13)
        right_knee = get_keypoint(keypoints, 14)
        left_ankle = get_keypoint(keypoints, 15)
        right_ankle = get_keypoint(keypoints, 16)
        left_shoulder = get_keypoint(keypoints, 5)
        right_shoulder = get_keypoint(keypoints, 6)
        left_elbow = get_keypoint(keypoints, 7)
        right_elbow = get_keypoint(keypoints, 8)

        # Use midpoints for ankles, knees, shoulders, and elbows
        mid_ankle = get_midpoint(left_ankle, right_ankle)
        mid_knee = get_midpoint(left_knee, right_knee)
        mid_shoulder = get_midpoint(left_shoulder, right_shoulder)
        mid_elbow = get_midpoint(left_elbow, right_elbow)

        # Append positions and angles to lists if all required keypoints are available
        if mid_ankle and mid_knee and mid_shoulder and mid_elbow:
            ankle_positions.append(mid_ankle)
            knee_positions.append(mid_knee)
            # Assume lead leg is forward
            lead_knee_positions.append(
                left_knee if left_knee[0] < right_knee[0] else right_knee)
            torso_angles.append(calculate_angle(
                mid_shoulder, mid_knee, mid_ankle))
            arm_positions.append(mid_elbow)

            # Criterion 1: Approach is done in 8 strides
            strides = detect_strides(ankle_positions)
            if len(strides) == 8:
                scoring['approach_8_strides'] = 1
                evaluation_frames[1].append(frame)

            # Criterion 2: Covers 4 contacts between hurdles
            # Lower threshold for contacts
            contacts = detect_strides(ankle_positions, stride_threshold=30)
            if len(contacts) == 4:
                scoring['4_contacts_between_hurdles'] = 1
                evaluation_frames[2].append(frame)

            # Criterion 3: Lead leg and torso alignment
            if check_lead_leg_and_torso(lead_knee_positions[-1], mid_ankle, torso_angles[-1], arm_positions[-1], hurdle_height):
                scoring['lead_leg_and_torso_alignment'] = 1
                evaluation_frames[3].append(frame)

            # Criterion 4: Second contact after hurdle involves a large stride with a visibly high knee
            if check_second_contact(ankle_positions, knee_positions):
                scoring['second_contact_after_hurdle'] = 1
                evaluation_frames[4].append(frame)

    return scoring, evaluation_frames
