import numpy as np
from scipy.signal import find_peaks

# ----------------- Helper Functions -----------------


def get_keypoint(keypoints, keypoint_index):
    """Retrieve a keypoint or return None if missing."""
    try:
        return keypoints[keypoint_index].tolist()
    except (IndexError, AttributeError):
        return None


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))


def calculate_angle(a, b, c):
    """Calculate the angle between three points in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle


def detect_strides(ankle_positions, stride_threshold=50):
    """Detect strides using ankle movement peaks."""
    if len(ankle_positions) < 2:
        return []

    distances = [calculate_distance(ankle_positions[i], ankle_positions[i-1])
                 for i in range(1, len(ankle_positions))]
    peaks, _ = find_peaks(distances, height=stride_threshold)
    return peaks.tolist()

# ----------------- Hurdling Criteria Evaluation Functions -----------------


def check_approach_strides(stride_indices, required_strides=8, side=''):
    """Check if the required number of approach strides is achieved."""
    if len(stride_indices) < required_strides:
        print(
            f"{side}: Insufficient strides detected (need at least {required_strides})")
        return False
    print(f"{side}: Approach completed in {len(stride_indices)} strides.")
    return True


def check_hurdle_contacts(leg_positions, stride_indices, required_contacts=4, side=''):
    """Check if the required number of hurdle contacts is achieved."""
    hurdle_contacts = 0
    for i in range(len(stride_indices) - 1):
        # Assuming y-coordinate is height
        if leg_positions[stride_indices[i]][1] < leg_positions[stride_indices[i + 1]][1]:
            hurdle_contacts += 1

    if hurdle_contacts < required_contacts:
        print(f"{side}: Insufficient hurdle contacts (need at least {required_contacts}, detected {hurdle_contacts})")
        return False

    print(f"{side}: Hurdle contacts detected = {hurdle_contacts} out of {len(stride_indices) - 1}")
    return True


def check_lead_leg_height(leg_positions, stride_indices, lead_leg_height_threshold=0.05, side=''):
    """Check if the lead leg passes above the hurdle."""
    lead_leg_passes_hurdle = all(
        leg_positions[i][1] > lead_leg_height_threshold for i in stride_indices
    )

    if not lead_leg_passes_hurdle:
        print(f"{side}: Lead leg does not pass above the hurdle.")
        return False

    print(f"{side}: Lead leg passes above the hurdle.")
    return True


def check_torso_movement(torso_positions, leg_positions, stride_indices, torso_movement_threshold=0.1, side=''):
    """Check if the torso moves toward the lead leg."""
    torso_movement = all(
        abs(torso_positions[i][0] - leg_positions[i][0]) < torso_movement_threshold for i in stride_indices
    )

    if not torso_movement:
        print(f"{side}: Torso does not move toward the lead leg.")
        return False

    print(f"{side}: Torso moves toward the lead leg.")
    return True


def check_high_knee_on_second_contact(leg_positions, stride_indices, high_knee_threshold=0.15, side=''):
    """Check if the second contact involves a high knee."""
    if len(stride_indices) < 2:
        print(f"{side}: Not enough strides to check for high knee on second contact.")
        return False

    # Assuming the second contact is the second stride
    second_contact_index = stride_indices[1]
    if leg_positions[second_contact_index][1] <= high_knee_threshold:
        print(f"{side}: Second contact does not involve a high knee.")
        return False

    print(f"{side}: Second contact involves a high knee.")
    return True


# ----------------- Main Evaluation Function -----------------


def evaluate_hurdling(player_coords):
    """
    Evaluate hurdling performance based on specified criteria.
    Returns:
        tuple: (scoring_dict, evaluation_frames_dict)
    """
    scoring = {
        'Approach takes 8 steps': 0,
        'Between the hurdles in 4 contacts': 0,
        'lead_leg_height': 0,
        'torso_movement': 0,
        'high_knee_on_second_contact': 0
    }

    evaluation_frames = {1: [], 2: [], 3: [], 4: [], 5: []}

    # Initialize trackers for both sides
    trackers = {
        'left': {
            'ankle': [], 'hip': [], 'torso': []
        },
        'right': {
            'ankle': [], 'hip': [], 'torso': []
        }
    }

    stride_indices = []

    for data in player_coords:
        frame = data['frame']
        keypoints = data['keypoints']

        # Get COCO-compliant keypoints
        left_hip = get_keypoint(keypoints, 11)  # Index 11: Left hip
        right_hip = get_keypoint(keypoints, 12)  # Index 12: Right hip
        left_ankle = get_keypoint(keypoints, 15)  # Index 15: Left ankle
        right_ankle = get_keypoint(keypoints, 16)  # Index 16: Right ankle

        # Approximate torso as the midpoint between hips
        torso = None
        if left_hip and right_hip:
            torso = [
                (left_hip[0] + right_hip[0]) / 2,  # x-coordinate
                (left_hip[1] + right_hip[1]) / 2   # y-coordinate
            ]

        # Update trackers with valid keypoints
        if left_ankle:
            trackers['left']['ankle'].append(left_ankle)
        if right_ankle:
            trackers['right']['ankle'].append(right_ankle)
        if left_hip:
            trackers['left']['hip'].append(left_hip)
        if right_hip:
            trackers['right']['hip'].append(right_hip)
        if torso:
            trackers['left']['torso'].append(torso)
            trackers['right']['torso'].append(torso)

        # Detect strides using right ankle (assuming right leg is lead leg)
        if right_ankle:
            stride_indices = detect_strides(trackers['right']['ankle'])

        # Evaluate criteria for each side
        for side in ['left', 'right']:
            # Criterion 1: Approach strides
            if check_approach_strides(stride_indices, side=side):
                scoring['Approach takes 8 steps'] = 1
                evaluation_frames[1].append(frame)

            # Criterion 2: Hurdle contacts
            if len(trackers[side]['ankle']) > 1:
                if check_hurdle_contacts(trackers[side]['ankle'], stride_indices, side=side):
                    scoring['Between the hurdles in 4 contacts'] = 1
                    evaluation_frames[2].append(frame)

            # Criterion 3: Lead leg height
            if len(trackers[side]['ankle']) > 0:
                if check_lead_leg_height(trackers[side]['ankle'], stride_indices, side=side):
                    scoring['lead_leg_height'] = 1
                    evaluation_frames[3].append(frame)

            # Criterion 4: Torso movement
            if len(trackers[side]['torso']) > 0 and len(trackers[side]['ankle']) > 0:
                if check_torso_movement(trackers[side]['torso'], trackers[side]['ankle'], stride_indices, side=side):
                    scoring['torso_movement'] = 1
                    evaluation_frames[4].append(frame)

            # Criterion 5: High knee on second contact
            if len(trackers[side]['ankle']) > 1:
                if check_high_knee_on_second_contact(trackers[side]['ankle'], stride_indices, side=side):
                    scoring['high_knee_on_second_contact'] = 1
                    evaluation_frames[5].append(frame)

    return scoring, evaluation_frames
