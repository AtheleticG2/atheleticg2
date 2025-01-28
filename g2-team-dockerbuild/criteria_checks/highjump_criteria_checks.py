import math
import logging
import cv2
from ultralytics import YOLO
import json
import numpy as np

# ------------- Logging --------------------

# # Configure logging at the top of your module
# logger = logging.getLogger(__name__)
# printvel(logging.DEBUG)  # Capture all levels of logs

# # Create handlers
# file_handler = logging.FileHandler('high_jump_debug.log')
# file_handler.setLevel(logging.DEBUG)

# # Console handler
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)

# # Create formatters and add them to handlers
# formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
# file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)

# # Add handlers to the logger
# logger.addHandler(file_handler)
# logger.addHandler(console_handler)

# ------------- Helper --------------------

def get_keypoint(kpts, idx):
    if idx < len(kpts):
        return kpts[idx]
    return None

def distance_2d(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def compute_angle_3pts(a, b, c):
    if a is None or b is None or c is None:
        print(f"compute_angle_3pts: One or more keypoints are None: a={a}, b={b}, c={c}")
        return None
    ax, ay = a
    bx, by = b
    cx, cy = c
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 < 1e-5 or mag2 < 1e-5:
        print(f"compute_angle_3pts: Magnitude too small: mag1={mag1}, mag2={mag2}")
        return None
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    try:
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        print(f"compute_angle_3pts: Computed angle={angle_deg:.2f} degrees at point b={b}")
        return angle_deg
    except ValueError:
        print(f"compute_angle_3pts: Invalid angle calculation with cos_angle={cos_angle}")
        return None

def compute_speed(center_curr, center_prev):
    if center_prev is None:
        print("compute_speed: Previous center is None.")
        return 0.0
    dx = center_curr[0] - center_prev[0]
    dy = center_curr[1] - center_prev[1]
    speed = math.hypot(dx, dy)
    print(f"compute_speed: Current center={center_curr}, Previous center={center_prev}, Speed={speed:.2f}")
    return speed

def get_bbox_center_xyxy(box):
    if box is None or len(box) != 4:
        print(f"get_bbox_center_xyxy: Invalid box={box}")
        return None
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    print(f"get_bbox_center_xyxy: Box={box}, Center=({cx:.2f}, {cy:.2f})")
    return (cx, cy)

# phase detection and segmentation 

def detect_phase_transitions(player_coords):
    if not player_coords:
        print("detect_phase_transitions: player_coords is empty.")
        return 0, 0, 0

    total = len(player_coords)
    one_fourth = total // 4
    half = total // 2
    three_fourths = (3 * total) // 4

    runup_end = one_fourth
    takeoff_end = half
    flight_end = three_fourths

    print(f"detect_phase_transitions: Phase transitions detected: runup_end={runup_end}, takeoff_end={takeoff_end}, flight_end={flight_end}")

    return runup_end, takeoff_end, flight_end

def segment_video_into_phases(player_coords, runup_end, takeoff_end, flight_end):

    runup_frames = player_coords[:runup_end]
    takeoff_frames = player_coords[runup_end:takeoff_end]
    flight_frames = player_coords[takeoff_end:flight_end]
    landing_frames = player_coords[flight_end:]

    print(f"segment_video_into_phases: Segments - runup_frames={len(runup_frames)}, takeoff_frames={len(takeoff_frames)}, flight_frames={len(flight_frames)}, landing_frames={len(landing_frames)}")

    return runup_frames, takeoff_frames, flight_frames, landing_frames


###################
#   CRITERION 1   #
###################

def is_running_tall(keypoints, shoulder_margin=30):

    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12

    left_shoulder = get_keypoint(keypoints, L_SHOULDER)
    right_shoulder = get_keypoint(keypoints, R_SHOULDER)
    left_hip = get_keypoint(keypoints, L_HIP)
    right_hip = get_keypoint(keypoints, R_HIP)

    if not (left_shoulder and right_shoulder and left_hip and right_hip):
        print(f"is_running_tall: Missing keypoints - left_shoulder={left_shoulder}, right_shoulder={right_shoulder}, left_hip={left_hip}, right_hip={right_hip}")
        return False

    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    hip_y = (left_hip[1] + right_hip[1]) / 2.0

    # If shoulders are significantly above hips, it's "running tall"
    result = (shoulder_y + shoulder_margin) < hip_y
    print(f"is_running_tall: shoulder_y={shoulder_y:.2f}, hip_y={hip_y:.2f}, Result={result}")
    return result

def is_accelerating(speed_history, min_increase_count=3):
    if len(speed_history) < min_increase_count + 1:
        print(f"is_accelerating: Not enough speed history data (required={min_increase_count +1}, available={len(speed_history)})")
        return False

    consecutive_increases = 0
    for i in range(1, len(speed_history)):
        if speed_history[i] > speed_history[i - 1]:
            consecutive_increases += 1
            print(f"is_accelerating: Speed increased from {speed_history[i -1]:.2f} to {speed_history[i]:.2f}, consecutive_increases={consecutive_increases}")
        else:
            print(f"is_accelerating: Speed did not increase from {speed_history[i -1]:.2f} to {speed_history[i]:.2f}")
            consecutive_increases = 0
        if consecutive_increases >= min_increase_count:
            print("is_accelerating: Acceleration criteria met.")
            return True

    print("is_accelerating: Acceleration criteria not met.")
    return False

def evaluate_runup_phase(runup_frames):
    partial_scoring = {
        'High Runup': 0
    }
    partial_eval_frames = {
        1: []
    }

    print(f"PHASE=Run-Up: Processing {len(runup_frames)} frames for Criterion 1.")

    speed_history = []
    initial_center = None
    center_previous = None

    for data in runup_frames:
        frame = data.get('frame', 0)
        kpts = data['keypoints']
        boxes = data.get('box', None)

        if boxes is not None:
            current_center = get_bbox_center_xyxy(boxes)
            if current_center is None:
                print(f"Frame {frame}: Invalid bounding box, skipping speed calculation.")
                continue

            if initial_center is None:
                initial_center = current_center
                center_previous = current_center
                print(f"Frame {frame}: Initial center set to {initial_center}")
            else:
                speed = compute_speed(current_center, center_previous)
                center_previous = current_center
                if speed > 0:
                    speed_history.append(speed)
                    print(f"Frame {frame}: Speed={speed:.2f} added to speed_history.")

            if is_accelerating(speed_history) and is_running_tall(kpts):
                partial_scoring['High Runup'] = 1
                partial_eval_frames[1].append(frame)
                print(f"Frame {frame}: Criterion 1 passed (accelerating and running tall).")
                break  #criterion met, no need to check further frames
        else:
            print(f"Frame {frame}: Missing bounding box for speed calculation.")

    print(f"Final scoring for Run-Up phase: {partial_scoring}")
    return partial_scoring, partial_eval_frames

##################
#   CRITERION 2  #
##################

def check_lean_in_curve(keypoints, angle_thresh=150):

    L_SHOULDER = 5
    R_SHOULDER = 6
    R_HIP = 12

    p_left_shoulder = get_keypoint(keypoints, L_SHOULDER)
    p_right_shoulder = get_keypoint(keypoints, R_SHOULDER)
    p_right_hip = get_keypoint(keypoints, R_HIP)

    angle_deg = compute_angle_3pts(p_left_shoulder, p_right_shoulder, p_right_hip)
    if angle_deg is None:
        # print(f"check_lean_in_curve: Unable to compute angle for keypoints - p_left_shoulder={p_left_shoulder}, p_right_shoulder={p_right_shoulder}, p_right_hip={p_right_hip}")
        return False

    result = (angle_deg < angle_thresh)
    # print(f"check_lean_in_curve: angle_deg={angle_deg:.2f}, angle_thresh={angle_thresh}, Result={result}")
    return result

def evaluate_leaning_phase(takeoff_frames):

    partial_scoring = {
        'Leaning during approach': 0
    }
    partial_eval_frames = {
        2: []
    }

    print(f"PHASE=Take-Off: Processing {len(takeoff_frames)} frames for Criterion 2.")

    for data in takeoff_frames:
        frame = data.get('frame', 0)
        kpts = data['keypoints']

        if check_lean_in_curve(kpts):
            partial_scoring['Leaning during approach'] = 1
            partial_eval_frames[2].append(frame)
            # print(f"Frame {frame}: Criterion 2 passed (leaning detected).")
            break  # Criterion met
    #     else:
    #         # print(f"Frame {frame}: Criterion 2 not met (no leaning).")

    # # print(f"Final scoring for Leaning phase: {partial_scoring}")
    return partial_scoring, partial_eval_frames

#############
#Criterion 3#
#############
def check_knee_lift_at_takeoff(keypoints, angle_thresh=120):
    L_HIP, L_KNEE, L_ANKLE = 11, 13, 15
    p_hip = get_keypoint(keypoints, L_HIP)
    p_knee = get_keypoint(keypoints, L_KNEE)
    p_ankle = get_keypoint(keypoints, L_ANKLE)

    # Log keypoint coordinates for verification
    # print(f"check_knee_lift_at_takeoff: Keypoints - Hip: {p_hip}, Knee: {p_knee}, Ankle: {p_ankle}")

    angle_deg = compute_angle_3pts(p_hip, p_knee, p_ankle)
    if angle_deg is None:
        # print(f"check_knee_lift_at_takeoff: Unable to compute angle for keypoints - Hip: {p_hip}, Knee: {p_knee}, Ankle: {p_ankle}")
        return False

    result = (angle_deg < angle_thresh)
    # print(f"check_knee_lift_at_takeoff: angle_deg={angle_deg:.2f}, angle_thresh={angle_thresh}, Result={result}")
    return result

def evaluate_takeoff_phase(takeoff_frames):
    partial_scoring = {
        'Full lift of the knee at take-off': 0
    }
    partial_eval_frames = {
        3: []
    }

    # print(f"PHASE=Take-Off: Processing {len(takeoff_frames)} frames for Criterion 3.")

    for data in takeoff_frames:
        frame = data.get('frame', 0)
        kpts = data['keypoints']

        if check_knee_lift_at_takeoff(kpts):
            partial_scoring['Full lift of the knee at take-off'] = 1
            partial_eval_frames[3].append(frame)
            # print(f"Frame {frame}: Criterion 3 passed (knee lift detected).")
            break  # Criterion met
        # else:
        #     # print(f"Frame {frame}: Criterion 3 not met (knee not fully lifted).")

    # print(f"Final scoring for Take-Off phase: {partial_scoring}")
    return partial_scoring, partial_eval_frames

#################
#  Criterion 4  #
#################

def check_hollow_back(keypoints, angle_thresh=160):
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14

    left_shoulder = get_keypoint(keypoints, L_SHOULDER)
    right_shoulder = get_keypoint(keypoints, R_SHOULDER)
    left_hip = get_keypoint(keypoints, L_HIP)
    right_hip = get_keypoint(keypoints, R_HIP)
    left_knee = get_keypoint(keypoints, L_KNEE)
    right_knee = get_keypoint(keypoints, R_KNEE)

    if not (left_shoulder and right_shoulder and left_hip and right_hip and left_knee and right_knee):
        # print(f"check_hollow_back: Missing keypoints - left_shoulder={left_shoulder}, right_shoulder={right_shoulder}, left_hip={left_hip}, right_hip={right_hip}, left_knee={left_knee}, right_knee={right_knee}")
        return False

    # Average shoulders
    shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    # Average hips
    hip_x = (left_hip[0] + right_hip[0]) / 2.0
    hip_y = (left_hip[1] + right_hip[1]) / 2.0
    # Average knees
    knee_x = (left_knee[0] + right_knee[0]) / 2.0
    knee_y = (left_knee[1] + right_knee[1]) / 2.0

    p_shoulder = (shoulder_x, shoulder_y)
    p_hip = (hip_x, hip_y)
    p_knee = (knee_x, knee_y)

    angle_deg = compute_angle_3pts(p_shoulder, p_hip, p_knee)
    if angle_deg is None:
        # print(f"check_hollow_back: Unable to compute angle_deg for keypoints.")
        return False

    result = (angle_deg > angle_thresh)
    # print(f"check_hollow_back: angle_deg={angle_deg:.2f}, angle_thresh={angle_thresh}, Result={result}")
    return result

def evaluate_flight_phase(flight_frames):
    partial_scoring = {
        'Clearing the bar with a hollow back': 0
    }
    partial_eval_frames = {
        4: []
    }

    # print(f"PHASE=Flight: Processing {len(flight_frames)} frames for Criterion 4.")

    for data in flight_frames:
        frame = data.get('frame', 0)
        kpts = data['keypoints']

        if check_hollow_back(kpts):
            partial_scoring['Clearing the bar with a hollow back'] = 1
            partial_eval_frames[4].append(frame)
            # print(f"Frame {frame}: Criterion 4 passed (hollow back detected).")
            break  # Criterion met
    #     else:
    #         # print(f"Frame {frame}: Criterion 4 not met (no hollow back).")

    # # print(f"Final scoring for Flight phase: {partial_scoring}")
    return partial_scoring, partial_eval_frames

#################
#  Criterion 5  #
#################

def check_l_shape_landing(keypoints, angle_range=(80, 100)):

    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12
    L_ANKLE, R_ANKLE = 15, 16

    left_shoulder = get_keypoint(keypoints, L_SHOULDER)
    right_shoulder = get_keypoint(keypoints, R_SHOULDER)
    left_hip = get_keypoint(keypoints, L_HIP)
    right_hip = get_keypoint(keypoints, R_HIP)
    left_ankle = get_keypoint(keypoints, L_ANKLE)
    right_ankle = get_keypoint(keypoints, R_ANKLE)

    if not (left_shoulder and right_shoulder and left_hip and right_hip and left_ankle and right_ankle):
        # print(f"check_l_shape_landing: Missing keypoints - left_shoulder={left_shoulder}, right_shoulder={right_shoulder}, left_hip={left_hip}, right_hip={right_hip}, left_ankle={left_ankle}, right_ankle={right_ankle}")
        return False

    #average shoulders
    shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    #average hips
    hip_x = (left_hip[0] + right_hip[0]) / 2.0
    hip_y = (left_hip[1] + right_hip[1]) / 2.0
    #average ankles
    ankle_x = (left_ankle[0] + right_ankle[0]) / 2.0
    ankle_y = (left_ankle[1] + right_ankle[1]) / 2.0

    p_shoulder = (shoulder_x, shoulder_y)
    p_hip = (hip_x, hip_y)
    p_ankle = (ankle_x, ankle_y)

    angle_deg = compute_angle_3pts(p_shoulder, p_hip, p_ankle)
    if angle_deg is None:
        # print(f"check_l_shape_landing: Unable to compute angle_deg for keypoints.")
        return False

    result = (angle_range[0] <= angle_deg <= angle_range[1])
    # print(f"check_l_shape_landing: angle_deg={angle_deg:.2f}, angle_range={angle_range}, Result={result}")
    return result

def evaluate_landing_phase(landing_frames):
    partial_scoring = {
        'Landing on the mat in an L and perpendicular to the bar': 0
    }
    partial_eval_frames = {
        5: []
    }

    # print(f"PHASE=Landing: Processing {len(landing_frames)} frames for Criterion 5.")

    for data in landing_frames:
        frame = data.get('frame', 0)
        kpts = data['keypoints']

        if check_l_shape_landing(kpts):
            partial_scoring['Landing on the mat in an L and perpendicular to the bar'] = 1
            partial_eval_frames[5].append(frame)
            # print(f"Frame {frame}: Criterion 5 passed (L-shaped landing detected).")
            break  # Criterion met
        # else:
            # print(f"Frame {frame}: Criterion 5 not met (no L-shaped landing).")

    # print(f"Final scoring for Landing phase: {partial_scoring}")
    return partial_scoring, partial_eval_frames

# ------------- Main -----------------

def evaluate_high_jump(player_coords):
    # print("Starting High Jump evaluation.")

    # 1) detect phase transitions
    runup_end, takeoff_end, flight_end = detect_phase_transitions(player_coords)
    # print(f"evaluate_high_jump: Phase transitions - runup_end={runup_end}, takeoff_end={takeoff_end}, flight_end={flight_end}")

    # 2) Segment frames by phase
    runup_frames, takeoff_frames, flight_frames, landing_frames = segment_video_into_phases(
        player_coords, runup_end, takeoff_end, flight_end
    )

    # 3) Evaluate each phase
    runup_scoring, runup_eval_frames = evaluate_runup_phase(runup_frames)
    leaning_scoring, leaning_eval_frames = evaluate_leaning_phase(takeoff_frames)
    takeoff_scoring, takeoff_eval_frames = evaluate_takeoff_phase(takeoff_frames)
    flight_scoring, flight_eval_frames = evaluate_flight_phase(flight_frames)
    landing_scoring, landing_eval_frames = evaluate_landing_phase(landing_frames)

    # 4) Merge results
    scoring = {
        'High Runup': runup_scoring.get('High Runup', 0),
        'Leaning during approach': leaning_scoring.get('Leaning during approach', 0),
        'Full lift of the knee at take-off': takeoff_scoring.get('Full lift of the knee at take-off', 0),
        'Clearing the bar with a hollow back': flight_scoring.get('Clearing the bar with a hollow back', 0),
        'Landing on the mat in an L and perpendicular to the bar': landing_scoring.get('Landing on the mat in an L and perpendicular to the bar', 0)
    }

    # Log merged scoring
    # print(f"evaluate_high_jump: Merged scoring - {scoring}")

    eval_frames = {
        1: runup_eval_frames.get(1, []),
        2: leaning_eval_frames.get(2, []),
        3: takeoff_eval_frames.get(3, []),
        4: flight_eval_frames.get(4, []),
        5: landing_eval_frames.get(5, [])
    }

    # print("High Jump evaluation completed.")
    # print(f"evaluate_high_jump: Evaluation frames - {eval_frames}")

    return scoring, eval_frames