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
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    angle_deg = math.degrees(math.acos(cos_angle))
    return angle_deg

def evaluate_discus_throw(player_coords):
    scoring = {
        'intro_swing_behind': 0,
        'jump_turn_initiated': 0,
        'jump_turn_center_circle': 0,
        'throw_off_low_to_high': 0,
        'discus_leaves_index_finger': 0
    }
    eval_frames = {1: [], 2: [], 3: [], 4: [], 5: []}

    for data in player_coords:
        frame = data['frame']
        kpts = data['keypoints']

        # Extract keypoints
        right_shoulder = get_keypoint(kpts, 6)
        right_hip = get_keypoint(kpts, 12)
        right_knee = get_keypoint(kpts, 14)
        right_ankle = get_keypoint(kpts, 16)
        left_hip = get_keypoint(kpts, 11)
        left_knee = get_keypoint(kpts, 13)
        left_ankle = get_keypoint(kpts, 15)
        right_wrist = get_keypoint(kpts, 10)

        # Compute angles and distances
        swing_angle = compute_angle_3pts(right_shoulder, right_hip, right_knee)
        jump_angle = compute_angle_3pts(right_hip, right_knee, right_ankle)
        throw_angle = compute_angle_3pts(right_knee, right_hip, left_knee)
        release_angle = compute_angle_3pts(right_shoulder, right_wrist, right_ankle)

        # Debug information
        print(f"[DEBUG Frame={frame}] swing_angle={swing_angle}, jump_angle={jump_angle}, "
              f"throw_angle={throw_angle}, release_angle={release_angle}")

        # Criterion 1: Introductory swing behind
        if right_wrist and right_shoulder and right_hip:
            swing_angle = compute_angle_3pts(right_wrist, right_shoulder, right_hip)
            if swing_angle is None:
                print(f"[DEBUG Frame {frame}] Missing keypoints for swing_angle calculation.")
            else:
                print(f"[DEBUG Frame {frame}] swing_angle={swing_angle}")
                # Ensure the arm is swung behind based on angle and position
                if swing_angle > 160 and right_wrist[0] < right_shoulder[0]:  # Check wrist is behind shoulder
                    scoring['intro_swing_behind'] = 1
                    eval_frames[1].append(frame)
                else:
                    print(f"[DEBUG Frame {frame}] Criterion 1 failed: swing_angle={swing_angle}, wrist={right_wrist[0]}, shoulder={right_shoulder[0]}")


        # Criterion 2: Jump turn initiated from ball of foot
        if right_ankle and right_knee and right_hip:
            jump_angle = compute_angle_3pts(right_hip, right_knee, right_ankle)
            if jump_angle is None:
                print(f"[DEBUG Frame {frame}] Missing keypoints for jump_angle calculation.")
            else:
                print(f"[DEBUG Frame {frame}] jump_angle={jump_angle}")
                if jump_angle > 80:  # Adjusted threshold
                    scoring['jump_turn_initiated'] = 1
                    eval_frames[2].append(frame)
                else:
                    print(f"[DEBUG Frame {frame}] Criterion 2 failed: jump_angle={jump_angle}")


        # Criterion 3: Jump turn near center of circle
        circle_center_x = 0.42
        threshold_distance = 0.05
        if right_ankle and left_ankle:
            mid_ankle_x = (right_ankle[0] + left_ankle[0]) / 2
            print(f"[DEBUG Frame {frame}] mid_ankle_x={mid_ankle_x}")
            if abs(mid_ankle_x - circle_center_x) < threshold_distance:
                print(f"[DEBUG Frame {frame}] Criterion 3 satisfied: mid_ankle_x={mid_ankle_x}")
                scoring['jump_turn_center_circle'] = 1
                eval_frames[3].append(frame)
            else:
                print(f"[DEBUG Frame {frame}] Criterion 3 failed: mid_ankle_x={mid_ankle_x}")


        # Criterion 4: Throw-off low to high
        if throw_angle is not None:
            print(f"[DEBUG Frame {frame}] Evaluating Criterion 4: throw_angle={throw_angle}")
            if 120 <= throw_angle <= 150:
                print(f"[DEBUG Frame {frame}] Criterion 4 satisfied: throw_angle={throw_angle}")
                scoring['throw_off_low_to_high'] = 1
                eval_frames[4].append(frame)
            else:
                print(f"[DEBUG Frame {frame}] Criterion 4 failed: throw_angle={throw_angle} not in range 120-150")
        else:
            print(f"[DEBUG Frame {frame}] Criterion 4 skipped: throw_angle is None")


        # Criterion 5: Discus leaves via index finger
        if release_angle is not None:
            print(f"[DEBUG Frame {frame}] Evaluating Criterion 5: release_angle={release_angle}")
            if release_angle > 150:
                print(f"[DEBUG Frame {frame}] Criterion 5 satisfied: release_angle={release_angle}")
                scoring['discus_leaves_index_finger'] = 1
                eval_frames[5].append(frame)
            else:
                print(f"[DEBUG Frame {frame}] Criterion 5 failed: release_angle={release_angle}")
        else:
            print(f"[DEBUG Frame {frame}] Criterion 5 skipped: release_angle is None")

    return scoring, eval_frames
