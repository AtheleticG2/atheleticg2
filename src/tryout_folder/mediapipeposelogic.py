import cv2
import mediapipe as mp
import numpy as np
import json

#initialize mediapipepose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

selected_point = None

def select_person(event, x, y, flags, param):
    global selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)

#calculate center of the torso
def get_torso_center(landmarks, width, height):
    left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height))
    right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width),
                      int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height))
    left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width),
                int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height))
    right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width),
                 int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height))

    #calculate average of the shoulder and hip points
    center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) // 4
    center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) // 4
    return (center_x, center_y)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle

    return angle

def get_point(landmarks, index, width, height):
    return (int(landmarks[index].x * width), int(landmarks[index].y * height))


#evaluation for sprint starting
def evaluate_sprint_start(landmarks, width, height):
    """Evaluate Sprint Starting Technique"""
    score = 0
    evaluation = []

    #define keypoints
    pelvis = get_point(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, width, height)
    shoulders = get_point(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, width, height)
    nose = get_point(landmarks, mp_pose.PoseLandmark.NOSE.value, width, height)
    back_leg = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value, width, height)

    # 1.pelvis is slightly higher than the shoulders
    if pelvis[1] < shoulders[1]:
        score += 1
        evaluation.append("Pelvis is slightly higher than shoulders")

    # 2.head is in line with the torso
    body_tilt_angle = calculate_angle(pelvis, nose, shoulders)
    if 70 <= body_tilt_angle <= 110:
        score += 1
        evaluation.append("Head is in line with torso")

    # 3.loss of balance toward the front
    left_hand = get_point(landmarks, mp_pose.PoseLandmark.LEFT_INDEX.value, width, height)
    right_hand = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_INDEX.value, width, height)

    average_hand_y = (left_hand[1] + right_hand[1]) // 2
    nose_to_hand_distance = abs(nose[1] - average_hand_y)

    if nose_to_hand_distance < 50:  # Nose is close to the hands
        score += 1
        evaluation.append("Nose is close to hands and ground")

    # 4.gaze directed obliquely forward
    nose_reference = (nose[0], nose[1] + 10)
    gaze_angle = calculate_angle(nose, nose_reference, shoulders)
    if 45 <= gaze_angle <= 70:
        score += 1
        evaluation.append("Gaze directed obliquely forward")

    # 5.first step with full extension of the back leg
    back_hip = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, width, height)
    back_knee = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value, width, height)
    back_ankle = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value, width, height)

    # Calculate the angle of the back leg
    back_leg_angle = calculate_angle(back_hip, back_knee, back_ankle)

    #check if the angle indicates full extension
    if back_leg_angle > 160:
        score += 1
        evaluation.append("First step with full extension of the back leg")

    return score, evaluation

#evaluation for sprint running
def evaluate_sprint_running(landmarks, width, height):
    score = 0
    evaluation = []

    #define keypoints
    left_ankle = get_point(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value, width, height)
    right_ankle = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value, width, height)
    left_knee = get_point(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value, width, height)
    right_knee = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value, width, height)
    left_hip = get_point(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, width, height)
    right_hip = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, width, height)

    # 1.walks on the ball of the foot
    left_ankle_angle = calculate_angle(left_knee, left_ankle, [left_ankle[0] + 10, left_ankle[1]])
    right_ankle_angle = calculate_angle(right_knee, right_ankle, [right_ankle[0] + 10, right_ankle[1]])
    if left_ankle_angle < 90 or right_ankle_angle < 90:
        score += 1
        evaluation.append("Walks on the ball of the foot")

    # 2.knees are high
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    if left_knee_angle < 90 or right_knee_angle < 90:
        score += 1
        evaluation.append("Knees are high")

    # 3.actively clawing at the ground
    left_hip_angle = calculate_angle(left_knee, left_hip, [left_hip[0] + 10, left_hip[1]])
    right_hip_angle = calculate_angle(right_knee, right_hip, [right_hip[0] + 10, right_hip[1]])
    if left_hip_angle < 90 or right_hip_angle < 90:
        score += 1
        evaluation.append("Actively clawing at the ground")

    # 4.arms at a 90° angle
    left_arm_angle = calculate_angle(
        get_point(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, width, height),
        get_point(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value, width, height),
        get_point(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value, width, height))
    right_arm_angle = calculate_angle(
        get_point(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, width, height),
        get_point(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value, width, height),
        get_point(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value, width, height))
    if 80 <= left_arm_angle <= 100 or 80 <= right_arm_angle <= 100:
        score += 1
        evaluation.append("Arms are at a 90° angle")

    # 5.body center of gravity tends forward
    body_tilt_angle = calculate_angle(left_hip, right_hip, [width // 2, height // 2])
    if body_tilt_angle > 70:
        score += 1
        evaluation.append("Body center of gravity tends forward")

    return score, evaluation

# evaluation for shot put
def evaluate_shot_put(landmarks, width, height):
    score = 0
    evaluation = []

    #eefine keypoints
    left_hip = get_point(mp_pose.PoseLandmark.LEFT_HIP.value)
    right_hip = get_point(mp_pose.PoseLandmark.RIGHT_HIP.value)
    left_knee = get_point(mp_pose.PoseLandmark.LEFT_KNEE.value)
    right_knee = get_point(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    left_ankle = get_point(mp_pose.PoseLandmark.LEFT_ANKLE.value)
    right_ankle = get_point(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    left_shoulder = get_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder = get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    left_elbow = get_point(mp_pose.PoseLandmark.LEFT_ELBOW.value)
    left_wrist = get_point(mp_pose.PoseLandmark.LEFT_WRIST.value)
    nose = get_point(mp_pose.PoseLandmark.NOSE.value)

    #1. glide phase with folded low leg, back to throwing direction
    back_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)  # Assuming left leg is folded
    shoulder_orientation = calculate_angle(left_shoulder, right_shoulder, [width, right_shoulder[1]])  # Horizontal reference
    if back_leg_angle < 90 and shoulder_orientation > 160:  # Shoulders nearly horizontal
        score += 1
        evaluation.append("Glide phase with folded low leg, back to throwing direction")

    #2. assisting leg pulled under the pelvis
    assisting_leg_height = abs(right_ankle[1] - right_hip[1])  # Vertical distance
    if assisting_leg_height < 50:  # Close to the pelvis
        score += 1
        evaluation.append("Assisting leg pulled under the pelvis")

    #3. sting leg with put down, butt leg is folded
    sting_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)  # Assuming right leg is the sting leg
    if sting_leg_angle < 90:  # Folded position
        score += 1
        evaluation.append("Sting leg with put down, butt leg is folded")

    #4. Push out punch, engage hip-torso before extending arm
    hip_torso_angle = calculate_angle(left_hip, nose, right_hip)  # Hip-torso rotation
    arm_extension_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)  # Arm extension
    if hip_torso_angle > 50 and arm_extension_angle > 150:  # Torso rotates before arm extends
        score += 1
        evaluation.append("Push out punch, engage hip-torso before extending arm")

    #5. Bullet remains in neck, arm extended at 45° angle
    bullet_in_neck = abs(left_wrist[1] - nose[1]) < 30  # Wrist close to neck
    release_angle = calculate_angle(left_shoulder, left_wrist, [left_wrist[0] + 10, left_wrist[1]])  # Arm release angle
    if bullet_in_neck and 40 <= release_angle <= 50:  # Arm at 45° angle
        score += 1
        evaluation.append("Bullet remains in neck, arm extended at release with 45° angle")

    return score, evaluation

def main():
    global selected_point

    #menu
    print("Select a sport for evaluation:")
    print("1. Sprint Starting")
    print("2. Sprint Running")
    print("3. Shot Put")
    sport_choice = int(input("Enter your choice with number: "))

    video_path = input("Enter the path to the video file: ")
    cap = cv2.VideoCapture(video_path)

    #video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter("evaluated_sport.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    cv2.namedWindow("select Person")
    cv2.setMouseCallback("select Person", select_person)

    selected_person = None
    highest_score = 0
    frame_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            torso_center = get_torso_center(landmarks, width, height)

            #if no person is selected, display torso centers for selection
            if selected_person is None:
                cv2.circle(frame, torso_center, 5, (0, 255, 0), -1)
                cv2.putText(frame, "Click on the correct person", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                #check if the user clicked and match the click to the torso center
                if selected_point:
                    distance = np.linalg.norm(np.array(torso_center) - np.array(selected_point))
                    if distance < 50:  #threshold for selection
                        selected_person = torso_center
                        print("Person selected!")
                        selected_point = None
            else:
                #track the selected person
                cv2.putText(frame, "Tracking selected person", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                #evaluate based on the selected sport
                if sport_choice == 1:
                    score, evaluation = evaluate_sprint_start(landmarks, width, height)
                elif sport_choice == 2:
                    score, evaluation = evaluate_sprint_running(landmarks, width, height)
                elif sport_choice == 3:
                    score, evaluation = evaluate_shot_put(landmarks, width, height)

                #update the highest score
                highest_score = max(highest_score, score)

                #save frame results
                frame_results.append({"frame": len(frame_results) + 1, "score": score, "evaluation": evaluation})

                #display results on the video
                cv2.putText(frame, f"Score: {score}/5", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset = 80
                for criteria in evaluation:
                    cv2.putText(frame, criteria, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 30

                #draw pose landmarks
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(66, 66, 245), thickness=2, circle_radius=2))

        #write frame to output video
        out.write(frame)

        #display frame
        cv2.imshow("Select Person", frame)

        #break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #save results to JSON
    evaluation_summary = {
        "video_path": video_path,
        "sport_choice": "Sprint Starting" if sport_choice == 1 else "Sprint Running" if sport_choice == 2 else "Shot Put",
        "highest_score": highest_score,
        "frame_results": frame_results
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation_summary, f, indent=4)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()