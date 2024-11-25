import cv2
import numpy as np
import mediapipe as mp

# --------------- MODULE 1: Utility Function to Calculate Angle ---------------
def calculate_angle(a, b, c):
    """
    Calculate the angle at joint b formed by points a, b, and c.

    Parameters:
        a (list): [x, y] coordinates of the first point.
        b (list): [x, y] coordinates of the second (joint) point.
        c (list): [x, y] coordinates of the third point.

    Returns:
        float: The angle (in degrees) at point b formed by points a, b, and c.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Vectors ba and bc
    ba = a - b
    bc = c - b

    # Calculate cosine of the angle
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    return angle


# --------------- MODULE 2: Squat Counting Function ---------------
def count_squats(video_path):
    """
    Analyze a video to count the number of squats performed.

    Parameters:
        video_path (str): Path to the video file to be processed.

    Returns:
        int: Total count of squats.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    squat_count = 0
    is_squat_down = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract relevant joints for squats
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate angle at the knee
            squat_angle = calculate_angle(hip, knee, ankle)

            # Squat logic: Detect "down" and "up" motions
            if squat_angle < 80 and not is_squat_down:  # Threshold for squat down
                is_squat_down = True
            elif squat_angle > 150 and is_squat_down:  # Threshold for squat up
                squat_count += 1
                is_squat_down = False

            # Visualize keypoints and angle on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw the angle at the knee
            cv2.putText(frame, f"Angle: {int(squat_angle)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.line(frame, (int(hip[0] * frame.shape[1]), int(hip[1] * frame.shape[0])),
                     (int(knee[0] * frame.shape[1]), int(knee[1] * frame.shape[0])), (0, 0, 255), 2)
            cv2.line(frame, (int(knee[0] * frame.shape[1]), int(knee[1] * frame.shape[0])),
                     (int(ankle[0] * frame.shape[1]), int(ankle[1] * frame.shape[0])), (0, 0, 255), 2)

            # Display counts and legend
            cv2.putText(frame, f"Squats: {squat_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Display the frame
        cv2.imshow('Squat Counter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return squat_count

# --------------- MODULE 3: Pushup Counting Function ---------------
def count_pushups(video_path):
    """
    Analyze a video to count the number of pushups performed.

    Parameters:
        video_path (str): Path to the video file to be processed.

    Returns:
        int: Total count of pushups.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    pushup_count = 0
    is_pushup_down = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract relevant joints for pushups
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle at the elbow
            pushup_angle = calculate_angle(shoulder, elbow, wrist)

            # Pushup logic: Detect "down" and "up" motions
            if pushup_angle < 90 and not is_pushup_down:  # Threshold for pushup down
                is_pushup_down = True
            elif pushup_angle > 160 and is_pushup_down:  # Threshold for pushup up
                pushup_count += 1
                is_pushup_down = False

            # Visualize keypoints and angle on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw the angle at the elbow
            cv2.putText(frame, f"Angle: {int(pushup_angle)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.line(frame, (int(shoulder[0] * frame.shape[1]), int(shoulder[1] * frame.shape[0])),
                     (int(elbow[0] * frame.shape[1]), int(elbow[1] * frame.shape[0])), (0, 0, 255), 2)
            cv2.line(frame, (int(elbow[0] * frame.shape[1]), int(elbow[1] * frame.shape[0])),
                     (int(wrist[0] * frame.shape[1]), int(wrist[1] * frame.shape[0])), (0, 0, 255), 2)

            # Display counts
            cv2.putText(frame, f"Pushups: {pushup_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Display the frame
        cv2.imshow('Pushup Counter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return pushup_count


# --------------- MODULE 4: Main Function ---------------
if __name__ == "__main__":
    exercise = "squats"
    video_path = f"{exercise}.mp4"  # Replace with your video file path    

    if exercise == "squats":
        squats = count_squats(video_path)
        print(f"Total Squats: {squats}")
    elif exercise == "pushups":
        pushups = count_pushups(video_path)
        print(f"Total Pushups: {pushups}")
    else:
        print("Unsupported exercise type!")
