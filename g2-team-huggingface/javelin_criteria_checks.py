import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import linregress


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


def fill_missing(data):
    """Linear interpolation for missing values (NaNs) in a 1D sequence."""
    arr = np.array(data)
    nans = np.isnan(arr)

    # If all values are NaN, return the original array
    if np.all(nans):
        return arr.tolist()

    # If no NaNs, return the original array
    if not np.any(nans):
        return arr.tolist()

    # Perform linear interpolation
    def indices(z): return z.nonzero()[0]
    arr[nans] = np.interp(indices(nans), indices(~nans), arr[~nans])
    return arr.tolist()


def merge_strides(strides):
    """Merge overlapping or adjacent stride intervals."""
    if not strides:
        return []

    sorted_strides = sorted(strides, key=lambda x: x[0])
    merged = [sorted_strides[0]]

    for current in sorted_strides[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 80:  # Allow 1 frame gap
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged


def extract_vertical(ankle_data, conf_threshold=0.2):
    """Extract Y-coordinates from ankle data with confidence check."""
    vertical_positions = []
    for kp in ankle_data:
        # Check if keypoint exists and has confidence > threshold
        if kp and len(kp) > 2 and kp[2] > conf_threshold:
            vertical_positions.append(kp[1])
        else:
            vertical_positions.append(np.nan)

    # If all values are NaN, return a default array (e.g., zeros)
    if all(np.isnan(v) for v in vertical_positions):
        # Default to zeros or another reasonable value
        return [0.0] * len(ankle_data)

    return vertical_positions


def detect_strides(left_ankle_data, right_ankle_data, freq=30,
                   min_stride_duration=0.01, max_stride_duration=0.8):
    """
    Detect strides using ankle vertical oscillations.
    Now accepts pre-collected left/right ankle data.
    """

    # Get vertical positions
    left_ankle = extract_vertical(left_ankle_data)
    right_ankle = extract_vertical(right_ankle_data)

    # Interpolate missing values
    left_ankle = fill_missing(left_ankle)
    right_ankle = fill_missing(right_ankle)

    # Create normalized differential signal
    norm_left = (left_ankle - np.nanmean(left_ankle)) / np.nanstd(left_ankle)
    norm_right = (right_ankle - np.nanmean(right_ankle)) / \
        np.nanstd(right_ankle)
    differential = norm_left - norm_right

    # If differential signal is invalid (e.g., all zeros), return empty strides
    if np.all(differential == 0) or np.isnan(differential).all():
        return []

    # filtering
    b, a = butter(3, [0.2, 7], fs=freq, btype='band')
    filtered = filtfilt(b, a, differential)

    # Find stride candidates
    peaks, _ = find_peaks(filtered, height=0.5,
                          distance=int(freq*min_stride_duration))
    valleys, _ = find_peaks(-filtered, height=0.5,
                            distance=int(freq*min_stride_duration))

    # Validate stride intervals
    strides = []
    for valley in valleys:
        prev_peaks = peaks[peaks < valley]
        if not prev_peaks:
            continue
        start = prev_peaks[-1]

        next_peaks = peaks[peaks > valley]
        if not next_peaks:
            continue
        end = next_peaks[0]

        duration = (end - start)/freq
        if min_stride_duration <= duration <= max_stride_duration:
            strides.append((start, end))

    return merge_strides(strides)

# ----------------- Criteria Evaluation Functions -----------------


# 1
def javelin_drawn_backward(shoulder_positions, wrist_positions, stride_indices, side,
                           last_n_strides=5, trend_threshold=-0.1, consistency_threshold=0.7):

    # Validate input data
    if not shoulder_positions or not wrist_positions or not stride_indices:
        print(f"{side}: Missing input data")
        return False

    # Handle insufficient strides
    valid_strides = min(len(stride_indices), last_n_strides)
    if valid_strides < 1:
        print(f"{side}: No valid strides available")
        return False

    # Extract continuous window for last N strides
    start_idx = max(0, stride_indices[-valid_strides] - 5)  # 5 frame buffer
    end_idx = stride_indices[-1] + 5  # 5 frame buffer

    # Collect valid data points with confidence check
    rel_positions = []
    for i in range(start_idx, min(end_idx+1, len(shoulder_positions))):
        try:
            # Skip frames with low confidence or missing data
            if (shoulder_positions[i][2] < 0.4 or
                    wrist_positions[i][2] < 0.4):
                continue

            rel_x = shoulder_positions[i][0] - wrist_positions[i][0]
            rel_positions.append(rel_x)
        except (IndexError, TypeError):
            continue

    if len(rel_positions) < 10:  # Minimum 10 valid frames
        print(f"{side}: Insufficient valid data ({len(rel_positions)} frames)")
        return False

    # Trend analysis with linear regression
    x = np.arange(len(rel_positions))
    slope, _, _, _, _ = linregress(x, rel_positions)

    # Consistency check
    backward_ratio = np.mean(np.array(rel_positions) > 0)

    print(f"{side}: Backward trend slope: {slope:.3f}")
    print(f"{side}: Backward frames ratio: {backward_ratio:.1%}")

    # Combined decision logic
    return (slope < trend_threshold and
            backward_ratio >= consistency_threshold)


# 2
def pelvis_rotation_and_javelin_drawn(hip_positions, shoulder_positions, wrist_positions, side,
                                      hip_rotation_threshold=-0.01, wrist_behind_threshold=0.04,
                                      pelvis_angle_threshold=80, vertical_alignment_threshold=0.1255,
                                      hip_stability_threshold=0.005, shoulder_stability_threshold=0.005,
                                      wrist_stability_threshold=0.01):

    if len(hip_positions) < 3 or len(shoulder_positions) < 3 or len(wrist_positions) < 3:
        print(
            f"{side}: Insufficient data for pelvis rotation and javelin check (need at least 3 frames)")
        return False

    # Calculate horizontal and vertical movement for hip, shoulder, and wrist
    hip_move_x = hip_positions[-1][0] - hip_positions[-2][0]
    # shoulder_move_x = shoulder_positions[-1][0] - shoulder_positions[-2][0]
    # wrist_move_x = wrist_positions[-1][0] - wrist_positions[-2][0]

    # Calculate the distance between wrist and shoulder
    wrist_behind_distance = shoulder_positions[-1][0] - wrist_positions[-1][0]

    # Calculate pelvis rotation angle (angle between hip, shoulder, and wrist)
    pelvis_angle = calculate_angle(
        hip_positions[-1], shoulder_positions[-1], wrist_positions[-1])

    # Calculate vertical alignment between pelvis and shoulder
    vertical_misalignment = abs(
        hip_positions[-1][1] - shoulder_positions[-1][1])

    # Calculate movement consistency (stability) over the last 3 frames
    hip_stability_x = abs(hip_positions[-1][0] - hip_positions[-3][0]) / 2
    shoulder_stability_x = abs(
        shoulder_positions[-1][0] - shoulder_positions[-3][0]) / 2
    wrist_stability_x = abs(
        wrist_positions[-1][0] - wrist_positions[-3][0]) / 2

    # Check for pelvis rotation (hip moves inward)
    if hip_move_x < hip_rotation_threshold:
        print(f"{side}: Hip movement ({hip_move_x:.2f}) is below rotation threshold ({hip_rotation_threshold:.2f})")
        return False

    # Check for javelin drawn back (wrist is behind shoulder)
    if wrist_behind_distance < wrist_behind_threshold:
        print(f"{side}: Wrist is not sufficiently behind shoulder ({wrist_behind_distance:.2f} < {wrist_behind_threshold:.2f})")
        return False

    # Check for pelvis rotation angle
    if pelvis_angle > pelvis_angle_threshold:
        print(f"{side}: Pelvis rotation angle ({pelvis_angle:.2f}) exceeds threshold ({pelvis_angle_threshold:.2f})")
        return False

    # Check for vertical alignment between pelvis and shoulder
    if vertical_misalignment > vertical_alignment_threshold:
        print(f"{side}: Vertical misalignment ({vertical_misalignment:.2f}) exceeds threshold ({vertical_alignment_threshold:.2f})")
        return False

    # Check for stability
    if hip_stability_x > hip_stability_threshold:
        print(f"{side}: Hip movement is not stable ({hip_stability_x:.2f})")
        return False
    if shoulder_stability_x > shoulder_stability_threshold:
        print(f"{side}: Shoulder movement is not stable ({shoulder_stability_x:.2f})")
        return False
    if wrist_stability_x > wrist_stability_threshold:
        print(f"{side}: Wrist movement is not stable ({wrist_stability_x:.2f})")
        return False

    return True


# 3
def impulse_step_executed(ankle_positions, knee_positions, hip_positions, side,
                          ankle_threshold=0.015, knee_threshold=0.1, hip_threshold=0.1,
                          stability_threshold=0.01):

    if len(ankle_positions) < 3 or len(knee_positions) < 3 or len(hip_positions) < 3:
        print(
            f"{side}: Insufficient data for impulse step check (need at least 3 frames)")
        return False

    # Calculate horizontal movement for ankle, knee, and hip
    ankle_move_x = ankle_positions[-1][0] - ankle_positions[-2][0]
    knee_move_x = knee_positions[-1][0] - knee_positions[-2][0]
    hip_move_x = hip_positions[-1][0] - hip_positions[-2][0]

    # Calculate movement consistency (stability) over the last 3 frames
    ankle_stability_x = abs(
        ankle_positions[-1][0] - ankle_positions[-3][0]) / 2
    knee_stability_x = abs(knee_positions[-1][0] - knee_positions[-3][0]) / 2
    hip_stability_x = abs(hip_positions[-1][0] - hip_positions[-3][0]) / 2

    # Debugging output
    print(f"{side}: Ankle movement (x) = {ankle_move_x:.2f}")
    print(f"{side}: Knee movement (x) = {knee_move_x:.2f}")
    print(f"{side}: Hip movement (x) = {hip_move_x:.2f}")
    print(f"{side}: Ankle stability (x) = {ankle_stability_x:.2f}")
    print(f"{side}: Knee stability (x) = {knee_stability_x:.2f}")
    print(f"{side}: Hip stability (x) = {hip_stability_x:.2f}")

    # Check for proper sequencing (ankle > knee > hip)
    if not (ankle_move_x > knee_move_x and knee_move_x > hip_move_x):
        print(f"{side}: Improper sequencing (ankle: {ankle_move_x:.2f}, knee: {knee_move_x:.2f}, hip: {hip_move_x:.2f})")
        return False

    # Check for minimum ankle movement
    if ankle_move_x < ankle_threshold:
        print(
            f"{side}: Ankle movement ({ankle_move_x:.2f}) is below threshold ({ankle_threshold:.2f})")
        return False

    # Check for maximum knee and hip movement
    if knee_move_x > knee_threshold:
        print(
            f"{side}: Knee movement ({knee_move_x:.2f}) exceeds threshold ({knee_threshold:.2f})")
        return False
    if hip_move_x > hip_threshold:
        print(
            f"{side}: Hip movement ({hip_move_x:.2f}) exceeds threshold ({hip_threshold:.2f})")
        return False

    # Check for stability
    if ankle_stability_x > stability_threshold:
        print(f"{side}: Ankle movement is not stable ({ankle_stability_x:.2f})")
        return False
    if knee_stability_x > stability_threshold:
        print(f"{side}: Knee movement is not stable ({knee_stability_x:.2f})")
        return False
    if hip_stability_x > stability_threshold:
        print(f"{side}: Hip movement is not stable ({hip_stability_x:.2f})")
        return False

    return True


# 4
def blocking_step_executed(ankle_positions, hip_positions, side,
                           ankle_threshold=0.0085, hip_threshold=0.01,
                           ankle_vertical_threshold=0.025, hip_vertical_threshold=0.15,
                           stability_threshold=0.01):

    if len(ankle_positions) < 3 or len(hip_positions) < 3:
        print(
            f"{side}: Insufficient data for blocking step check (need at least 3 frames)")
        return False

    # Calculate horizontal and vertical movement for ankle and hip
    ankle_move_x = abs(ankle_positions[-1][0] - ankle_positions[-2][0])
    ankle_move_y = abs(ankle_positions[-1][1] - ankle_positions[-2][1])
    hip_move_x = abs(hip_positions[-1][0] - hip_positions[-2][0])
    hip_move_y = abs(hip_positions[-1][1] - hip_positions[-2][1])

    # Calculate movement consistency (stability) over the last 3 frames
    ankle_stability_x = abs(
        ankle_positions[-1][0] - ankle_positions[-3][0]) / 2
    ankle_stability_y = abs(
        ankle_positions[-1][1] - ankle_positions[-3][1]) / 2
    hip_stability_x = abs(hip_positions[-1][0] - hip_positions[-3][0]) / 2
    hip_stability_y = abs(hip_positions[-1][1] - hip_positions[-3][1]) / 2

    # # Debugging output
    # print(f"{side}: Ankle movement (x, y) = ({ankle_move_x:.2f}, {ankle_move_y:.2f})")
    # print(f"{side}: Hip movement (x, y) = ({hip_move_x:.2f}, {hip_move_y:.2f})")
    # print(f"{side}: Ankle stability (x, y) = ({ankle_stability_x:.2f}, {ankle_stability_y:.2f})")
    # print(f"{side}: Hip stability (x, y) = ({hip_stability_x:.2f}, {hip_stability_y:.2f})")

    # Stricter ankle movement check (horizontal and vertical)
    if ankle_move_x > ankle_threshold:
        print(f"{side}: Ankle horizontal movement ({ankle_move_x:.2f}) exceeds threshold ({ankle_threshold:.2f})")
        return False
    if ankle_move_y > ankle_vertical_threshold:
        print(f"{side}: Ankle vertical movement ({ankle_move_y:.2f}) exceeds threshold ({ankle_vertical_threshold:.2f})")
        return False

    # Stricter hip movement check (horizontal and vertical)
    if hip_move_x < hip_threshold:
        print(f"{side}: Hip horizontal movement ({hip_move_x:.2f}) is below threshold ({hip_threshold:.2f})")
        return False
    if hip_move_y > hip_vertical_threshold:
        print(f"{side}: Hip vertical movement ({hip_move_y:.2f}) exceeds threshold ({hip_vertical_threshold:.2f})")
        return False

    # Stability check (movement consistency over the last 3 frames)
    if ankle_stability_x > stability_threshold or ankle_stability_y > stability_threshold:
        print(f"{side}: Ankle movement is not stable (x: {ankle_stability_x:.2f}, y: {ankle_stability_y:.2f})")
        return False
    if hip_stability_x > stability_threshold or hip_stability_y > stability_threshold:
        print(
            f"{side}: Hip movement is not stable (x: {hip_stability_x:.2f}, y: {hip_stability_y:.2f})")
        return False

    return True


# 5
def throw_initiated(hip_positions, shoulder_positions, wrist_positions, side,
                    torso_angle_threshold=100, wrist_height_relative_threshold=-0.15,
                    progressive_movement_threshold=0.01):

    if len(hip_positions) < 2 or len(shoulder_positions) < 2 or len(wrist_positions) < 2:
        print(f"{side}: Insufficient data for throw initiation check")
        return False

    hip_move = hip_positions[-1][0] - hip_positions[-2][0]
    shoulder_move = shoulder_positions[-1][0] - shoulder_positions[-2][0]
    wrist_move = wrist_positions[-1][0] - wrist_positions[-2][0]
    torso_angle = calculate_angle(
        hip_positions[-1], shoulder_positions[-1], wrist_positions[-1])
    wrist_height_relative = wrist_positions[-1][1] - shoulder_positions[-1][1]

    if len(hip_positions) >= 3:
        progressive_hip_move = hip_positions[-1][0] - hip_positions[-3][0]
        progressive_shoulder_move = shoulder_positions[-1][0] - \
            shoulder_positions[-3][0]
        progressive_wrist_move = wrist_positions[-1][0] - \
            wrist_positions[-3][0]
    else:
        progressive_hip_move = hip_move
        progressive_shoulder_move = shoulder_move
        progressive_wrist_move = wrist_move

    if torso_angle < torso_angle_threshold:
        return False
    if wrist_height_relative < wrist_height_relative_threshold:
        return False
    if (abs(progressive_hip_move) < progressive_movement_threshold or
        abs(progressive_shoulder_move) < progressive_movement_threshold or
            abs(progressive_wrist_move) < progressive_movement_threshold):
        return False
    return True


# ----------------- Main Evaluation Function -----------------
def evaluate_javelin_throw(player_coords):
    """
    Evaluate javelin throw technique based on 5 criteria.
    Returns:
        tuple: (scoring_dict, evaluation_frames_dict)
    """
    scoring = {
        'Javelin drawn backwards': 0,
        'Pelvis rotated and javelin is fully drawn backwards': 0,
        'Impulse step exeecuted': 0,
        'Blocking step executed': 0,
        'Throw initiated': 0
    }

    evaluation_frames = {1: [], 2: [], 3: [], 4: [], 5: []}

    # Initialize trackers for both sides
    trackers = {
        'left': {
            'shoulder': [], 'wrist': [], 'hip': [],
            'knee': [], 'ankle': []
        },
        'right': {
            'shoulder': [], 'wrist': [], 'hip': [],
            'knee': [], 'ankle': []
        }
    }

    stride_indices = []

    for data in player_coords:
        frame = data['frame']
        keypoints = data['keypoints']

        # Get COCO-compliant keypoints (indices 0-16)
        current_points = {
            'left': {
                'shoulder': get_keypoint(keypoints, 5),
                'wrist': get_keypoint(keypoints, 9),
                'hip': get_keypoint(keypoints, 11),
                'knee': get_keypoint(keypoints, 13),
                'ankle': get_keypoint(keypoints, 15)
            },
            'right': {
                'shoulder': get_keypoint(keypoints, 6),
                'wrist': get_keypoint(keypoints, 10),
                'hip': get_keypoint(keypoints, 12),
                'knee': get_keypoint(keypoints, 14),
                'ankle': get_keypoint(keypoints, 16)
            }
        }

        # Update trackers with valid keypoints
        for side in ['left', 'right']:
            for joint in ['shoulder', 'wrist', 'hip', 'knee', 'ankle']:
                if current_points[side][joint]:
                    trackers[side][joint].append(current_points[side][joint])

        # Detect strides using right ankle (assuming right-handed throw)
        if current_points['left']['ankle'] and current_points['right']['ankle']:
            stride_indices = detect_strides(
                trackers['left']['ankle'], trackers['right']['ankle'])  # Pass right ankle data

        # Evaluate criteria for each side

        for side in ['left', 'right']:

            # Criterion 1: Only check at end of stride sequence
            if frame == len(player_coords) - 1:  # Last frame check
                if javelin_drawn_backward(trackers[side]['shoulder'],
                                          trackers[side]['wrist'],
                                          stride_indices, side):
                    scoring['Javelin drawn backwards'] = 1
                    # Mark all frames in last 5 strides
                    if len(stride_indices) >= 5:
                        evaluation_frames[1] = list(range(
                            stride_indices[-5],
                            stride_indices[-1] + 1
                        ))

            # Criterion 2: Pelvis rotation and javelin drawn back
            if (len(trackers[side]['hip']) > 1 and
                len(trackers[side]['shoulder']) > 1 and
                    len(trackers[side]['wrist']) > 0):
                if pelvis_rotation_and_javelin_drawn(trackers[side]['hip'],
                                                     trackers[side]['shoulder'],
                                                     trackers[side]['wrist'], side):
                    scoring['Pelvis rotated and javelin is fully drawn backwards'] = 1
                    evaluation_frames[2].append(frame)

            # Criterion 3: Impulse step executed
            if (len(trackers[side]['ankle']) > 1 and
                len(trackers[side]['knee']) > 1 and
                    len(trackers[side]['hip']) > 1):
                if impulse_step_executed(trackers[side]['ankle'],
                                         trackers[side]['knee'],
                                         trackers[side]['hip'], side):
                    scoring['Impulse step exeecuted'] = 1
                    evaluation_frames[3].append(frame)

            # Criterion 4: Blocking step executed
            if len(trackers[side]['ankle']) > 1 and len(trackers[side]['hip']) > 1:
                if blocking_step_executed(trackers[side]['ankle'],
                                          trackers[side]['hip'], side):
                    scoring['Blocking step executed'] = 1
                    evaluation_frames[4].append(frame)

            # Criterion 5: Throw initiated through hips and torso
            if (len(trackers[side]['hip']) > 1 and
                len(trackers[side]['shoulder']) > 1 and
                    len(trackers[side]['wrist']) > 1):
                if throw_initiated(trackers[side]['hip'],
                                   trackers[side]['shoulder'],
                                   trackers[side]['wrist'], side):
                    scoring['Throw initiated'] = 1
                    evaluation_frames[5].append(frame)

    return scoring, evaluation_frames
