import math
import logging


# ------------- helper geometry functions --------------------

def get_keypoint(kpts, idx):
    if idx < len(kpts):
        return kpts[idx]
    return None

# def distance_2d(p1, p2):
#     if p1 is None or p2 is None:
#         return None
#     return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
def distance_2d(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def compute_angle_3pts(a, b, c):
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
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    try:
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    except ValueError:
        # logger.debug(f"Invalid angle calculation with cos_angle={cos_angle}")
        return None

def shoulder_orientation(left_shoulder, right_shoulder):
    if left_shoulder is None or right_shoulder is None:
        return None
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    return math.degrees(math.atan2(dy, dx))

# ------------- phase detection and segmentation --------------------

def detect_phase_transitions(player_coords):
    if not player_coords:
        # logger.debug("player_coords is empty.")
        return 0, 0

    total = len(player_coords)
    one_third = total // 3
    two_thirds = (2 * total) // 3

    preparation_end_index = one_third
    transition_end_index = two_thirds

    # logger.debug(f"preparation_end_index={preparation_end_index}, transition_end_index={transition_end_index}")

    return preparation_end_index, transition_end_index

def segment_video_into_phases(player_coords, preparation_end_index, transition_end_index):
    preparation_phase_frames = player_coords[:preparation_end_index]
    transition_phase_frames = player_coords[preparation_end_index:transition_end_index]
    release_phase_frames = player_coords[transition_end_index:]

    # # logger.debug(f"Segments: preparation_frames={len(preparation_phase_frames)}, "
    #              f"transition_frames={len(transition_phase_frames)}, "
    #              f"release_frames={len(release_phase_frames)}")

    return preparation_phase_frames, transition_phase_frames, release_phase_frames

# ------------- criterion checks by phase --------------------
def evaluate_preparation_phase(preparation_frames):
    partial_scoring = {
        'Initiate the glide phase from a folded low leg, facing away from the throwing direction.': 0
    }
    partial_eval_frames = {
        1: []
    }

    # logger.debug(f"PHASE=Preparation: Processing {len(preparation_frames)} frames for Criterion 1.")

    for idx, data in enumerate(preparation_frames):
        frame = data.get('frame', idx)
        kpts = data['keypoints']

        # logger.debug(f"PHASE=Preparation Frame={frame}: Checking Criterion 1...")

        left_hip = get_keypoint(kpts, 11)
        right_hip = get_keypoint(kpts, 12)
        left_knee = get_keypoint(kpts, 13)
        right_knee = get_keypoint(kpts, 14)
        left_ankle = get_keypoint(kpts, 15)
        right_ankle = get_keypoint(kpts, 16)
        left_shoulder = get_keypoint(kpts, 5)
        right_shoulder = get_keypoint(kpts, 6)

        #llog extracted keypoints
        # # logger.debug(f"Frame {frame}: left_hip={left_hip}, right_hip={right_hip}, "
        #              f"left_knee={left_knee}, right_knee={right_knee}, "
        #              f"left_ankle={left_ankle}, right_ankle={right_ankle}, "
        #              f"left_shoulder={left_shoulder}, right_shoulder={right_shoulder}")

        #criterion 1: glide phase initiated from a folded low leg, with back to the throwing direction
        if (left_hip and right_hip and left_knee and right_knee and
            left_ankle and right_ankle and left_shoulder and right_shoulder):

            #angle at the right knee
            right_knee_angle = compute_angle_3pts(left_hip, right_knee, right_ankle)
            left_knee_angle = compute_angle_3pts(right_hip, left_knee, left_ankle)

            #shoulder orientation
            orient_angle = shoulder_orientation(left_shoulder, right_shoulder)

            # logger.debug(f"Frame {frame}: right_knee_angle={right_knee_angle:.2f} degrees, "
                        #  f"left_knee_angle={left_knee_angle:.2f} degrees, "
                        #  f"shoulder_orientation={orient_angle:.2f} degrees")

            #define thresholds
            if (right_knee_angle is not None and left_knee_angle is not None and orient_angle is not None):
                if (right_knee_angle < 180 and left_knee_angle < 180 and
                    (orient_angle > 70 or orient_angle < -70)):
                    partial_scoring['Initiate the glide phase from a folded low leg, facing away from the throwing direction.'] = 1
                    partial_eval_frames[1].append(frame)
                    # logger.debug(f"Frame {frame}: Criterion 1 passed.")
                # else:
                    # logger.debug(f"Frame {frame}: Criterion 1 failed: "
        #                          f"right_knee_angle={right_knee_angle:.2f}, "
        #                          f"left_knee_angle={left_knee_angle:.2f}, "
        #                          f"shoulder_orientation={orient_angle:.2f}")
        # else:
            # logger.debug(f"Frame {frame}: Missing keypoints required for Criterion 1 evaluation.")

    # logger.debug(f"Final scoring for Preparation phase: {partial_scoring}")
    return partial_scoring, partial_eval_frames

def evaluate_transition_phase(transition_frames):
    partial_scoring = {
        'Execute a flat hop to pull the assisting leg under the pelvis.': 0,
        'Sting leg is down while keeping the butt leg folded after the flat hop': 0
    }
    partial_eval_frames = {
        2: [],
        3: []
    }

    # logger.debug(f"PHASE=Transition: Processing {len(transition_frames)} frames for Criteria 2 & 3.")

    for idx, data in enumerate(transition_frames):
        frame = data.get('frame', idx + len(transition_frames))
        kpts = data['keypoints']

        # logger.debug(f"PHASE=Transition Frame={frame}: Checking Criteria 2 & 3...")

        left_hip = get_keypoint(kpts, 11)
        right_hip = get_keypoint(kpts, 12)
        left_knee = get_keypoint(kpts, 13)
        right_knee = get_keypoint(kpts, 14)
        left_ankle = get_keypoint(kpts, 15)
        right_ankle = get_keypoint(kpts, 16)

        # logger.debug(f"Frame {frame}: left_hip={left_hip}, right_hip={right_hip}, "
                    #  f"left_knee={left_knee}, right_knee={right_knee}, "
                    #  f"left_ankle={left_ankle}, right_ankle={right_ankle}")

        #criterion 2: using a flat hop, assisting leg pulled under the pelvis
        if (left_hip and left_knee and left_ankle):
            #assuming left leg is assisting leg
            left_knee_angle = compute_angle_3pts(left_hip, left_knee, left_ankle)
            # logger.debug(f"Frame {frame}: left_knee_angle={left_knee_angle:.2f} degrees")

            if left_knee_angle is not None and left_knee_angle < 160: 
                partial_scoring['Execute a flat hop to pull the assisting leg under the pelvis.'] = 1
                partial_eval_frames[2].append(frame)
      
        #criterion 3: stiff leg with put down, butt leg is still folded after the flat hop
        if (right_hip and right_knee and right_ankle and left_knee and left_hip):
            #assuming right leg is stiff leg
            right_knee_angle = compute_angle_3pts(right_hip, right_knee, right_ankle)
            left_knee_angle = compute_angle_3pts(left_hip, left_knee, left_ankle)


            if (right_knee_angle is not None and left_knee_angle is not None):
                if (right_knee_angle > 140 and left_knee_angle < 160):
                    partial_scoring['Sting leg is down while keeping the butt leg folded after the flat hop'] = 1
                    partial_eval_frames[3].append(frame)
                    # logger.debug(f"Frame {frame}: Criterion 3 passed.")
                
    return partial_scoring, partial_eval_frames

def evaluate_release_phase(release_frames):
    partial_scoring = {
        'Execute a push-out punch, engaging the hip-torso before arm extension': 0,
        'Keep the ball near the neck until arm extension, pushing at a 45° angle.': 0
    }
    partial_eval_frames = {
        4: [],
        5: []
    }

    # logger.debug(f"PHASE=Release: Processing {len(release_frames)} frames for Criteria 4 & 5.")

    for idx, data in enumerate(release_frames):
        frame = data.get('frame', idx + len(release_frames))
        kpts = data['keypoints']

        # logger.debug(f"PHASE=Release Frame={frame}: Checking Criteria 4 & 5...")

        left_shoulder = get_keypoint(kpts, 5)
        right_shoulder = get_keypoint(kpts, 6)
        left_elbow = get_keypoint(kpts, 7)
        right_elbow = get_keypoint(kpts, 8)
        left_wrist = get_keypoint(kpts, 9)
        right_wrist = get_keypoint(kpts, 10)
        left_hip = get_keypoint(kpts, 11)
        right_hip = get_keypoint(kpts, 12)
        nose = get_keypoint(kpts, 0) 


        #criterion 4:push out punch, then engage the hip-torso before extending the arm
        if (left_shoulder and right_shoulder and left_elbow and right_elbow and
            left_hip and right_hip):

            left_elbow_angle = compute_angle_3pts(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = compute_angle_3pts(right_shoulder, right_elbow, right_wrist)
            shoulder_to_hip_angle = compute_angle_3pts(left_shoulder, left_hip, right_shoulder)

        
            #thresholds based on criteria
            if (left_elbow_angle is not None and right_elbow_angle is not None and
                shoulder_to_hip_angle is not None):

                #'N/A' if angle is None
                left_elbow_str = f"{left_elbow_angle:.2f}" if left_elbow_angle is not None else "N/A"
                right_elbow_str = f"{right_elbow_angle:.2f}" if right_elbow_angle is not None else "N/A"
                shoulder_to_hip_str = f"{shoulder_to_hip_angle:.2f}" if shoulder_to_hip_angle is not None else "N/A"

                

                if (left_elbow_angle > 160 and right_elbow_angle > 160 and
                    (shoulder_to_hip_angle > 30 or shoulder_to_hip_angle < -30)): 
                    partial_scoring['Execute a push-out punch, engaging the hip-torso before arm extension'] = 1
                    partial_eval_frames[4].append(frame)
                    # logger.debug(f"Frame {frame}: Criterion 4 passed.")
               

        #criterion 5:ball remains in neck until arm is extended at release. ball is pushed at a 45° angle
        if (nose and right_wrist and right_shoulder):

            #ball position
            dist_wr_nose = distance_2d(right_wrist, nose)

            #arm extension angle (shoulder to wrist)
            arm_release_angle = None
            if right_shoulder and right_wrist:
                dx_arm = right_wrist[0] - right_shoulder[0]
                dy_arm = right_wrist[1] - right_shoulder[1]
                arm_release_angle = math.degrees(math.atan2(dy_arm, dx_arm))

            if (dist_wr_nose is not None and arm_release_angle is not None):
                dist_wr_nose_str = f"{dist_wr_nose:.2f}" if dist_wr_nose is not None else "N/A"
                arm_release_angle_str = f"{arm_release_angle:.2f}" if arm_release_angle is not None else "N/A"

              

                if (dist_wr_nose < 50 and 30 <= arm_release_angle <= 60):
                    partial_scoring['Keep the ball near the neck until arm extension, pushing at a 45° angle'] = 1
                    partial_eval_frames[5].append(frame)
                    # logger.debug(f"Frame {frame}: Criterion 5 passed.")
       
    return partial_scoring, partial_eval_frames

# ------------- main evaluation --------------------
def evaluate_shot_put(player_coords):
    # logger.info("Starting Shot Put evaluation.")

    # 1) detect phase transitions
    preparation_end_index, transition_end_index = detect_phase_transitions(player_coords)
   
    # 2) segment frames by phase
    preparation_frames, transition_frames, release_frames = segment_video_into_phases(
        player_coords, preparation_end_index, transition_end_index
    )

    # 3) evaluate each phase
    preparation_scoring, preparation_eval_frames = evaluate_preparation_phase(preparation_frames)
    transition_scoring, transition_eval_frames = evaluate_transition_phase(transition_frames)
    release_scoring, release_eval_frames = evaluate_release_phase(release_frames)

    # 4) merge results
    scoring = {
        'Initiate the glide phase from a folded low leg, facing away from the throwing direction': preparation_scoring.get('Initiate the glide phase from a folded low leg, facing away from the throwing direction', 0),
        'Execute a flat hop to pull the assisting leg under the pelvis': transition_scoring.get('Execute a flat hop to pull the assisting leg under the pelvis', 0),
        'Sting leg is down while keeping the butt leg folded after the flat hop': transition_scoring.get('Sting leg is down while keeping the butt leg folded after the flat hop', 0),
        'Execute a push-out punch, engaging the hip-torso before arm extension': release_scoring.get('Execute a push-out punch, engaging the hip-torso before arm extension', 0),
        'Keep the ball near the neck until arm extension, pushing at a 45° angle': release_scoring.get('Keep the ball near the neck until arm extension, pushing at a 45° angle', 0)
    }

    # log merged scoring
    # logger.debug(f"Merged scoring: {scoring}")

    eval_frames = {
        1: preparation_eval_frames.get(1, []),
        2: transition_eval_frames.get(2, []),
        3: transition_eval_frames.get(3, []),
        4: release_eval_frames.get(4, []),
        5: release_eval_frames.get(5, [])
    }

    # logger.info("Shot Put evaluation completed.")
    return scoring, eval_frames
