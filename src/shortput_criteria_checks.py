import math

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
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    # Clamp to avoid floating precision issues:
    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    angle_deg = math.degrees(math.acos(cos_angle))
    return angle_deg

def shoulder_orientation(left_shoulder, right_shoulder):
    """
    Returns angle in degrees from left_shoulder to right_shoulder
    relative to horizontal. ~0 => right_shoulder is directly right,
    ~180 => left_shoulder on right, etc. Returns None if invalid.
    """
    if left_shoulder is None or right_shoulder is None:
        return None
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    return math.degrees(math.atan2(dy, dx))

def evaluate_shot_put(player_coords):

    scoring = {
        'criterion_1': 0,
        'criterion_2': 0,
        'criterion_3': 0,
        'criterion_4': 0,
        'criterion_5': 0
    }
    eval_frames = {1: [], 2: [], 3: [], 4: [], 5: []}

    for data in player_coords:
        frame = data['frame']
        kpts  = data['keypoints']

        #extract keypoints
        left_hip = get_keypoint(kpts, 11)
        right_hip = get_keypoint(kpts, 12)
        left_knee = get_keypoint(kpts, 13)
        right_knee = get_keypoint(kpts, 14)
        left_ankle = get_keypoint(kpts, 15)
        right_ankle = get_keypoint(kpts, 16)
        left_shoulder = get_keypoint(kpts, 5)
        right_shoulder = get_keypoint(kpts, 6)
        right_elbow = get_keypoint(kpts, 8)
        right_wrist = get_keypoint(kpts, 10)
        right_ear = get_keypoint(kpts, 4)

        #angles/distances
        rk_angle = compute_angle_3pts(right_hip, right_knee, right_ankle)
        lk_angle = compute_angle_3pts(left_hip, left_knee, left_ankle)
        re_angle = compute_angle_3pts(right_shoulder, right_elbow, right_wrist)
        orient_angle = shoulder_orientation(left_shoulder, right_shoulder)
        dist_wr_ear = distance_2d(right_wrist, right_ear)
        arm_release_angle = None
        if right_shoulder and right_wrist:
            dx_arm = right_wrist[0] - right_shoulder[0]
            dy_arm = right_wrist[1] - right_shoulder[1]
            arm_release_angle = math.degrees(math.atan2(dy_arm, dx_arm))

        #debug
        print(f"[DEBUG Frame={frame}] "
              f"rkA={rk_angle}, lkA={lk_angle}, reA={re_angle}, "
              f"orient={orient_angle}, dist_wr_ear={dist_wr_ear}, "
              f"arm_release={arm_release_angle}")

        # Criterion 1: Glide phase
        if rk_angle is not None and orient_angle is not None:
            if rk_angle < 120 and 130 <= abs(orient_angle) <= 220:
                scoring['criterion_1'] = 1
                eval_frames[1].append(frame)

        # Criterion 2: Assisting leg under pelvis
        if left_ankle and left_hip:
            dist_ankle_hip = distance_2d(left_ankle, left_hip)
            if left_ankle[1] < left_hip[1] and dist_ankle_hip < 150:
                scoring['criterion_2'] = 1
                eval_frames[2].append(frame)

        # Criterion 3: Stiff right leg, folded left leg
        if rk_angle is not None and lk_angle is not None:
            if rk_angle > 140 and lk_angle < 140:
                scoring['criterion_3'] = 1
                eval_frames[3].append(frame)

        # Criterion 4: Push/punch, engage hip-torso
        if re_angle is not None and right_hip and right_shoulder:
            if re_angle > 140 and right_hip[0] > right_shoulder[0]:
                scoring['criterion_4'] = 1
                eval_frames[4].append(frame)

        # Criterion 5: Ball near ear until ~45° release
        if dist_wr_ear is not None and re_angle is not None and arm_release_angle is not None:
            if dist_wr_ear < 100 and re_angle > 140 and 30 <= arm_release_angle <= 60:
                scoring['criterion_5'] = 1
                eval_frames[5].append(frame)

    return scoring, eval_frames
