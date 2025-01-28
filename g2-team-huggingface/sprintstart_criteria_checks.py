import ultralytics
from ultralytics import YOLO

import numpy as np

# Function to calculate angles between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
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

def get_player_coords(player_id: int, results, box_incl:bool = False, keypoints_as_list:bool = False):
    player_coords = []

    # Iterate through each result with an associated frame index
    for frame_index, result in enumerate(results):
        if hasattr(result, 'boxes') and hasattr(result, 'keypoints'):
            boxes = result.boxes
            keypoints = result.keypoints

            if boxes is not None and keypoints is not None:
                tracking_ids = boxes.id
                kp_xyn = keypoints.xyn
                boxes_xy = boxes.xyxy

                if tracking_ids is not None:
                    for track_id, kp, box in zip(tracking_ids, kp_xyn, boxes_xy):
                        if int(track_id) == player_id:
                            if keypoints_as_list:
                                kp = kp.tolist()
                            # Append keypoints along with the frame number
                            if box_incl:
                                box = [int(coord) for coord in box]
                                player_coords.append({'frame': frame_index, 'keypoints': kp, 'box': box})
                            else:
                                player_coords.append({'frame': frame_index, 'keypoints': kp})
    return player_coords



def sprint_start_crit_5(left_knee_angles, right_knee_angles, extended_threshold=100, contracted_threshold=95):
    """
    Check if one leg is almost fully extended while the other is contracted.
    """
    if len(left_knee_angles) <1 or len(right_knee_angles) <1:
        return False

    if (left_knee_angles[-1] > extended_threshold and right_knee_angles[-1] < contracted_threshold) or \
      (right_knee_angles[-1] > extended_threshold and left_knee_angles[-1] < contracted_threshold):
        return True

    return False


def evaluate_sprint_start(player_coords):
    scoring = {
        'Pelvis slightly higher than shoulders': 0, 
        'Head in line with torso': 0,
        'Legs push off forcefully': 0, 
        'Gaze directed towards the ground': 0,
        'Back leg fully extended': 0
    }

    # Initialize lists to track knee angles over time
    left_knee_angles = []
    right_knee_angles = []

    evaluation_frames = {1:[],2:[],3:[],4:[],5:[]}

    for data in player_coords:
        frame = data['frame']
        keypoints = data['keypoints']
        left_hip = get_keypoint(keypoints, 11)
        right_hip = get_keypoint(keypoints, 12)
        left_shoulder = get_keypoint(keypoints, 5)
        right_shoulder = get_keypoint(keypoints, 6)
        left_ear= get_keypoint(keypoints, 3)
        right_ear = get_keypoint(keypoints, 4)
        nose = get_keypoint(keypoints, 0)
        left_knee = get_keypoint(keypoints, 13)
        left_ankle = get_keypoint(keypoints, 15)
        right_knee = get_keypoint(keypoints, 14)
        right_ankle = get_keypoint(keypoints, 16)

        if not (left_hip and right_hip and left_shoulder and left_knee and left_ankle and right_knee and right_ankle):
            continue

        mid_hip = get_midpoint(left_hip, right_hip)
        mid_ear = get_midpoint(left_ear, right_ear)
        mid_shoulder = get_midpoint(left_shoulder, right_shoulder)

        # Criterion 1: pelvis slightly higher than the shoulders
        if mid_hip[1] < mid_shoulder[1]:
            scoring['Pelvis slightly higher than shoulders'] = 1
            evaluation_frames[1].append(frame)

        # Criterion 2: Head aligned with torso
        body_tilt_angle = calculate_angle(mid_hip, mid_ear, mid_shoulder)
        if 0 <= body_tilt_angle <= 4:
            scoring['Head in line with torso'] = 1
            evaluation_frames[2].append(frame)


        # Track knee angles over time
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        left_knee_angles.append(left_knee_angle)
        right_knee_angles.append(right_knee_angle)

        # Criterion 3: Legs push off forcefully
        if len(left_knee_angles) > 1 and len(right_knee_angles) > 1:
          
          left_angle_change = left_knee_angles[-1] - left_knee_angles[-2]
          right_angle_change = right_knee_angles[-1] - right_knee_angles[-2]
          
        # Check if either one of the legs are almost fully extended with a significant change in angle in a few frames
          if left_knee_angles[-1] > 170 and left_angle_change > 25 or \
            right_knee_angles[-1] > 170 and right_angle_change > 25:
              
              scoring['Legs push off forcefully'] = 1
              evaluation_frames[3].append(frame)

        # Criterion 5: One leg extended, the other contracted i.e. full extension of the back leg
        if sprint_start_crit_5(left_knee_angles, right_knee_angles):
            scoring['Back leg fully extended'] = 1
            evaluation_frames[5].append(frame)

        # Criterion 4: Gaze directed towards the ground
        if nose[1] > mid_shoulder[1]:  # nose is below the shoulder, and body leans forward (origin is located on the top left corner)
          scoring['Gaze directed towards the ground'] = 1
          evaluation_frames[4].append(frame)


    return scoring, evaluation_frames
