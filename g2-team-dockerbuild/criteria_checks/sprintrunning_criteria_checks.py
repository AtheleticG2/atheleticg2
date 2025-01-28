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



def is_running_on_balls_of_feet(ankle, knee, threshold=0):
    """
    Check if the player is running on the balls of their feet by comparing the ankle and knee positions.
    The ankle should be higher than the knee by a certain threshold.
    """
    # reference_length = ((ankle[0] - knee[0])**2 + (ankle[1] - knee[1])**2)**0.5
    # Calculate the vertical distance between the ankle and knee
    vertical_distance = (ankle[1] - knee[1]) 
    
    if vertical_distance < threshold:
      return True

    return False



def center_of_gravity_leans_forward(hip_positions, shoulder_positions, angle_threshold=10):
    """
    Check if the center of gravity leans forward by analyzing the torso angle.
    """
    
    if len(hip_positions) < 1 or len(shoulder_positions) < 1:
        return False

    # Calculate the torso angle
    torso_angle = calculate_angle(hip_positions[-1], shoulder_positions[-1], [shoulder_positions[-1][0], hip_positions[-1][1]])

    # Check if the torso is leaning forward
    if torso_angle > angle_threshold:
        return True

    return False


def sprint_running_crit_1(hip, knee, vertical_threshold=0.15):
    # Check if the knee and hip are aligned vertically with some leeway
    vertical_distance = (knee[1] - hip[1])
    if vertical_distance > vertical_threshold:
        return True

    return False

def is_actively_clawing_at_ground(ankle_grounded, knee_grounded, hip_grounded,knee_other_leg, hip_other_leg, angle_threshold=85):
    """
    Check if the player is actively clawing at the ground by evaluating the angle at the ankle of the grounded foot
    with respect to the knee of the other leg at the hip.
    """
    # Calculate the angle at the ankle of the grounded foot
    clawing_angle = calculate_angle(hip_other_leg, knee_other_leg, knee_grounded)
    knee_angle = calculate_angle(ankle_grounded, knee_grounded, hip_grounded) # check for full extension of the grounded leg

    if angle_threshold <= clawing_angle <= (180 - angle_threshold) and knee_angle>=170:
        return True

    return False

def evaluate_sprint_running(player_coords):
    scoring = {
        'Knees are lifted high': 0,
        'Runs on balls of feet': 0,
        'Arms at a 90º angle': 0,
        'Center of gravity leans forward': 0,
        'Actively clawing at the ground':0 # New criterion
    }

    # Initialize lists to track knee angles over time
    left_knee_angles = []
    right_knee_angles = []

    # Initialize lists to track ankle and hip positions over time
    left_ankle_positions = []
    right_ankle_positions = []
    left_hip_positions = []
    right_hip_positions = []
    left_shoulder_positions = []
    right_shoulder_positions = []

    evaluation_frames = {1:[],2:[],3:[],4:[],5:[]} # for validation
   

    for data in player_coords:
        frame = data['frame']
        keypoints = data['keypoints']
        left_shoulder = get_keypoint(keypoints, 5)
        right_shoulder = get_keypoint(keypoints, 6)
        left_hip = get_keypoint(keypoints, 11)
        right_hip = get_keypoint(keypoints, 12)
        left_knee = get_keypoint(keypoints, 13)
        left_ankle = get_keypoint(keypoints, 15)
        right_knee = get_keypoint(keypoints, 14)
        right_ankle = get_keypoint(keypoints, 16)
        left_wrist = get_keypoint(keypoints, 9)
        left_elbow = get_keypoint(keypoints, 7)
        right_wrist = get_keypoint(keypoints, 10)
        right_elbow = get_keypoint(keypoints, 8)


  
        if not (left_ankle and right_ankle and left_hip and right_hip):
            continue

        # Track positions
        left_ankle_positions.append(left_ankle)
        right_ankle_positions.append(right_ankle)
        left_shoulder_positions.append(left_shoulder)
        right_shoulder_positions.append(right_shoulder)

        left_hip_positions.append(left_hip)
        right_hip_positions.append(right_hip)

        # Track knee angles over time
        left_pelvis_angles = calculate_angle(left_shoulder,left_hip, left_knee)
        right_pelvis_angles = calculate_angle(right_shoulder,right_hip, right_knee)

        left_knee_angles.append(left_pelvis_angles)
        right_knee_angles.append(right_pelvis_angles)

        # Criterion 1: Knees are high (knee lifted relative to hip)
        if sprint_running_crit_1(left_hip, left_knee) or \
           sprint_running_crit_1(right_hip, right_knee):
            scoring['Knees are lifted high'] = 1
            evaluation_frames[1].append(frame)


        # Criterion 2: Runs on balls of feet
        if is_running_on_balls_of_feet(left_ankle, left_knee) or \
           is_running_on_balls_of_feet(right_ankle, right_knee):
            scoring['Runs on balls of feet'] = 1
            evaluation_frames[2].append(frame)


        # Criterion 3: Arms at 90 degrees
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        if 79 <= left_arm_angle <= 105 and 79 <= right_arm_angle <= 105:
            scoring['Arms at a 90º angle'] = 1
            evaluation_frames[3].append(frame)

        # Criterion 4: Center of gravity leans forward - check if hips lean more forward compared to the feet
        if center_of_gravity_leans_forward(right_hip_positions, right_shoulder_positions):
            scoring['Center of gravity leans forward'] = 1
            evaluation_frames[4].append(frame)
        
        # Criterion 5: Actively clawing at the ground
        if is_actively_clawing_at_ground(left_ankle,left_knee, left_hip,right_knee, right_hip) or \
          is_actively_clawing_at_ground(right_ankle,right_knee, right_hip,left_knee, left_hip):
            scoring['Actively clawing at the ground'] = 1
            evaluation_frames[5].append(frame)

    return scoring, evaluation_frames

            