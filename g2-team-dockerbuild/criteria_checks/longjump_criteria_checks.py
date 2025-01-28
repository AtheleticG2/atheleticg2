import math
import logging
import cv2
import os


#############################
#    FRAME SAVING CONFIG    #
#############################
# CRITERIA_FRAMES_DIR = "criteria_frames"
# os.makedirs(CRITERIA_FRAMES_DIR, exist_ok=True)

# def save_frame(image, frame, criterion_id):
#     filename = os.path.join(CRITERIA_FRAMES_DIR, f"criterion_{criterion_id}_frame_{frame}.jpg")
#     if image is None:
#         # logger.error(f"Frame {frame}: No valid image provided for Criterion {criterion_id}.")
#         return None
#     success = cv2.imwrite(filename, image)
#     if success:
#         # logger.info(f"Saved frame {frame} for Criterion {criterion_id} at {filename}")
#         return filename
#     else:
#         # logger.error(f"Failed to save frame {frame} for Criterion {criterion_id}.")
#         return None

#############################
#    HELPER FUNCTIONS       #
#############################

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
        # logger.debug(f"Invalid angle calculation with cos_angle={cos_angle}")
        return None

#############################
#    CRITERION LOGIC        #
#############################

#################
#  Criterion 1  #
#################

# 1. The approach is done with acceleration, without slowing down/tripping before the push-off
def is_accelerating(speed_history, min_increase_count=3, speed_threshold=10.0):
    if len(speed_history) < min_increase_count + 1:
        # logger.debug("Not enough data points in speed history to evaluate acceleration.")
        return False

    # Track consecutive increases
    consecutive_increases = 0
    for i in range(1, len(speed_history)):
        # Check if speed is increasing and exceeds the threshold
        if speed_history[i] > speed_history[i - 1] and speed_history[i] > speed_threshold:
            consecutive_increases += 1
        
        else:
            consecutive_increases = 0

        # Return true if the condition is met
        if consecutive_increases >= min_increase_count:
            # logger.debug(f"Acceleration detected: {consecutive_increases} consecutive increases.")
            return True

    # logger.debug(f"No sufficient consecutive increases detected. Maximum consecutive increases: {consecutive_increases}")
    return False



#################
#  Criterion 2  #
#################

# Board region boundaries (adjust based on video quality)
BOARD_REGION = (195, 230, 350, 400)  # (xmin, xmax, ymin, ymax)

def foot_on_board(kpts, board_region):
    RIGHT_ANKLE = 16
    foot_x = kpts[RIGHT_ANKLE][0]
    foot_y = kpts[RIGHT_ANKLE][1]

    xmin, xmax, ymin, ymax = board_region
    if (xmin <= foot_x <= xmax) and (ymin <= foot_y <= ymax):
        # logger.debug(f"Foot on board: ({foot_x}, {foot_y}) within {board_region}.")
        return True
    # logger.debug(f"Foot not on board: ({foot_x}, {foot_y}) outside {board_region}.")
    return False

def check_not_looking_down(kpts):
    NOSE = 0
    L_EYE = 1
    R_EYE = 2
    if kpts[NOSE][1] < kpts[L_EYE][1] and kpts[NOSE][1] < kpts[R_EYE][1]:
        # logger.debug("Athlete is not looking down.")
        return True
    # logger.debug("Athlete is looking down.")
    return False

def evaluate_criterion2(kpts):
    foot_ok = foot_on_board(kpts, BOARD_REGION)
    head_ok = check_not_looking_down(kpts)

    if foot_ok and head_ok:
        # logger.debug("Criterion 2 passed: Foot on board and not looking down.")
        return True
    # logger.debug("Criterion 2 failed.")
    return False

#################
#  Criterion 3  #
#################

# 3. Push-off foot is flat on the ground and body's center of gravity is above it (not on heel and leaning back)
def check_foot_flat_and_com_over_foot(kpts):
    R_ANKLE = 16
    R_KNEE  = 14
    R_HIP   = 12
    # Calculate angle at the knee
    p_ankle = (kpts[R_ANKLE][0], kpts[R_ANKLE][1])
    p_knee  = (kpts[R_KNEE][0],  kpts[R_KNEE][1])
    p_hip   = (kpts[R_HIP][0],   kpts[R_HIP][1])

    angle_deg = compute_angle_3pts(p_ankle, p_knee, p_hip)
    if angle_deg is None:
        # logger.debug("Unable to compute foot angle due to missing keypoints.")
        return False

    # Check if angle is straight enough
    foot_flat_ok = (angle_deg > 165)
    # logger.debug(f"Foot angle at knee: {angle_deg:.2f} degrees. Flat: {foot_flat_ok}")

    # Check if center of mass is above foot (simplified by aligning x-coordinates)
    COM_x = p_hip[0]  # Assuming center of mass is at hip
    foot_x = p_ankle[0]
    com_over_foot = abs(COM_x - foot_x) < 10  # Threshold for alignment

    # logger.debug(f"Center of mass alignment: COM_x={COM_x:.2f}, Foot_x={foot_x:.2f}, Aligned: {com_over_foot}")

    return foot_flat_ok and com_over_foot

#################
#  Criterion 4  #
#################

# 4. Repulsive leg not retracted – "knightly stance" early
def check_repulsive_leg_not_retracted(kpts):
    """
    Ensure the repulsive leg is not retracted too early by checking the left knee angle.
    """
    L_HIP   = 11
    L_KNEE  = 13
    L_ANKLE = 15
    p_hip   = (kpts[L_HIP][0],   kpts[L_HIP][1])
    p_knee  = (kpts[L_KNEE][0],  kpts[L_KNEE][1])
    p_ankle = (kpts[L_ANKLE][0], kpts[L_ANKLE][1])

    angle_deg = compute_angle_3pts(p_hip, p_knee, p_ankle)
    if angle_deg is None:
        # logger.debug("Unable to compute repulsive leg angle due to missing keypoints.")
        return False

    # Check if angle is sufficiently extended
    repulsive_leg_ok = (angle_deg > 120)
    # logger.debug(f"Repulsive leg angle: {angle_deg:.2f} degrees. Not retracted: {repulsive_leg_ok}")

    return repulsive_leg_ok

#################
#  Criterion 5  #
#################

# 5. Landing is done using sliding technique
def check_sliding_landing(kpts):
    """
    Verify if the athlete lands with a sliding technique by checking the alignment of shoulders, hips, and ankles.
    """
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP           = 11, 12
    L_ANKLE, R_ANKLE       = 15, 16

    # Calculate average positions
    sh_x = (kpts[L_SHOULDER][0] + kpts[R_SHOULDER][0]) / 2.0
    sh_y = (kpts[L_SHOULDER][1] + kpts[R_SHOULDER][1]) / 2.0
    hip_x = (kpts[L_HIP][0] + kpts[R_HIP][0]) / 2.0
    hip_y = (kpts[L_HIP][1] + kpts[R_HIP][1]) / 2.0
    an_x = (kpts[L_ANKLE][0] + kpts[R_ANKLE][0]) / 2.0
    an_y = (kpts[L_ANKLE][1] + kpts[R_ANKLE][1]) / 2.0

    p_shoulder = (sh_x, sh_y)
    p_hip      = (hip_x, hip_y)
    p_ankle    = (an_x, an_y)

    angle_deg = compute_angle_3pts(p_shoulder, p_hip, p_ankle)
    # Check if angle is approximately 90 degrees for sliding posture
    sliding_ok = (80 <= angle_deg <= 100)
    # logger.debug(f"Sliding landing angle: {angle_deg:.2f} degrees. Sliding technique: {sliding_ok}")

    return sliding_ok

#############################
#      MAIN EVALUATION      #
#############################

def evaluate_long_jump(player_coords):
    # Initialize scoring dictionary for 5 criteria
    scoring = {
        'The approach maintains acceleration without slowing down before the push-off': 0, 
        'Not looking at the push-off bar; ensure the push-off foot is flat on the ground.': 0,
        'Keep the center of gravity above the push-off foot, avoiding heel contact and leaning back': 0, 
        'Repulsive leg not retracted': 0, 
        'Sliding landing': 0
    }

    # Track saved frames for each criterion
    saved_frames = {criterion: None for criterion in scoring.keys()}

    # Initialize tracking variables
    speed_history = []
    evaluation_frames = {1: [], 2: [], 3: [], 4: [], 5: []}
    runup_started = False
    initial_center = None
    center_previous = None
    DISPLACEMENT_THRESHOLD = 80.0 
    for data in player_coords:
        frame = data['frame']
        kpts = data['keypoints']
        boxes = data.get('box') 
        frame_image = data.get('frame_image')  # Optional frame image for saving

        if not boxes or not kpts:
            # logger.debug(f"Frame {frame}: Missing bounding box or keypoints.")
            continue

        # Get bounding box center
        x1, y1, x2, y2 = map(int, boxes)
        current_center = get_bbox_center_xyxy([x1, y1, x2, y2])
        # logger.debug(f"Frame {frame}: Bounding box center at {current_center}.")

        # Criterion 1: Accelerating run-up (sequence-based)
        if not runup_started:
            if initial_center is None:
                initial_center = current_center
                # logger.debug(f"Frame {frame}: Initial center set to {initial_center}.")
            else:
                disp = distance_2d(current_center, initial_center)
                if disp > DISPLACEMENT_THRESHOLD:
                    runup_started = True
                    center_previous = current_center
                    speed_history.clear()
                    scoring['The approach maintains acceleration without slowing down before the push-off'] = 1
                    evaluation_frames[1].append(frame)
                    # logger.debug(f"Frame {frame}: Run-up started.")
        else:
            # Update speed history
            speed = compute_speed(current_center, center_previous)
            center_previous = current_center
            if speed > 0:
                speed_history.append(speed)
                # logger.debug(f"Frame {frame}: Current speed: {speed:.2f} pixels/frame.")

            if not scoring['The approach maintains acceleration without slowing down before the push-off']:
                if is_accelerating(speed_history, min_increase_count=3, speed_threshold=5.0):
                    scoring['The approach maintains acceleration without slowing down before the push-off'] = 1
                    evaluation_frames[1].append(frame)
                    # logger.debug(f"Frame {frame}: Criterion 1 passed (Accelerating run-up).")

        # Criterion 2: Foot on board and not looking down (single-frame)
        if not scoring['Not looking at the push-off bar; ensure the push-off foot is flat on the ground.']:
            if foot_on_board(kpts, BOARD_REGION) and check_not_looking_down(kpts):
                scoring['Not looking at the push-off bar; ensure the push-off foot is flat on the ground.'] = 1
                evaluation_frames[2].append(frame)
                # logger.debug(f"Frame {frame}: Criterion 2 passed (Foot on board & Not looking down).")
                
        # Criterion 3: Foot flat and COM above foot (single-frame)
        if not scoring['Keep the center of gravity above the push-off foot, avoiding heel contact and leaning back']:
            if check_foot_flat_and_com_over_foot(kpts):
                scoring['Keep the center of gravity above the push-off foot, avoiding heel contact and leaning back'] = 1
                evaluation_frames[3].append(frame)
                # logger.debug(f"Frame {frame}: Criterion 3 passed (Foot flat & COM above).")
               

        # Criterion 4: Repulsive leg not retracted (single-frame)
        if not scoring['Repulsive leg not retracted']:
            if check_repulsive_leg_not_retracted(kpts):
                scoring['Repulsive leg not retracted'] = 1
                evaluation_frames[4].append(frame)
                # logger.debug(f"Frame {frame}: Criterion 4 passed (Repulsive leg not retracted).")
               

        # Criterion 5: Sliding landing (single-frame)
        if not scoring['Sliding landing']:
            if check_sliding_landing(kpts):
                scoring['Sliding landing'] = 1
                evaluation_frames[5].append(frame)
                # logger.debug(f"Frame {frame}: Criterion 5 passed (Sliding landing).")
              
    return scoring, saved_frames




#############################
#     HELPER FUNCTIONS      #
#############################

def get_bbox_center_xyxy(box):
    """Returns the center (cx, cy) of the bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy)

def compute_speed(center_curr, center_prev):
    """Approximate speed in pixels/frame between consecutive bounding box centers."""
    if center_prev is None:
        return 0.0
    dx = center_curr[0] - center_prev[0]
    dy = center_curr[1] - center_prev[1]
    return math.hypot(dx, dy)
