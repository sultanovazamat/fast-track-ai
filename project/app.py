import gradio as gr
import cv2
import numpy as np
import mediapipe as mp

# --------------- MODULE 1: Utility Function to Calculate Angle ---------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    return angle

# --------------- MODULE 2: Squat Counting Function (Refactored for Live Updates) ---------------
def count_squats_live(video):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video)
    squat_count = 0
    is_squat_down = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            squat_angle = calculate_angle(hip, knee, ankle)

            if squat_angle < 80 and not is_squat_down:
                is_squat_down = True
            elif squat_angle > 150 and is_squat_down:
                squat_count += 1
                is_squat_down = False

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Draw the angle at the knee
            cv2.putText(frame, f"Angle: {int(squat_angle)}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.line(frame, (int(hip[0] * frame.shape[1]), int(hip[1] * frame.shape[0])),
                     (int(knee[0] * frame.shape[1]), int(knee[1] * frame.shape[0])), (0, 0, 255), 2)
            cv2.line(frame, (int(knee[0] * frame.shape[1]), int(knee[1] * frame.shape[0])),
                     (int(ankle[0] * frame.shape[1]), int(ankle[1] * frame.shape[0])), (0, 0, 255), 2)

            # Display counts and legend
            cv2.putText(frame, f"Squats: {squat_count}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap.release()

# --------------- MODULE 3: Pushup Counting Function (Refactored for Live Updates) ---------------
def count_pushups_live(video):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video)
    pushup_count = 0
    is_pushup_down = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            pushup_angle = calculate_angle(shoulder, elbow, wrist)

            if pushup_angle < 90 and not is_pushup_down:
                is_pushup_down = True
            elif pushup_angle > 160 and is_pushup_down:
                pushup_count += 1
                is_pushup_down = False

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Draw the angle at the elbow
            cv2.putText(frame, f"Angle: {int(pushup_angle)}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.line(frame, (int(shoulder[0] * frame.shape[1]), int(shoulder[1] * frame.shape[0])),
                     (int(elbow[0] * frame.shape[1]), int(elbow[1] * frame.shape[0])), (0, 0, 255), 2)
            cv2.line(frame, (int(elbow[0] * frame.shape[1]), int(elbow[1] * frame.shape[0])),
                     (int(wrist[0] * frame.shape[1]), int(wrist[1] * frame.shape[0])), (0, 0, 255), 2)

            # Display counts
            cv2.putText(frame, f"Pushups: {pushup_count}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap.release()

# --------------- MODULE 4: Gradio Interface ---------------
def process_video_live(video, exercise):
    if exercise == "squats":
        generator = count_squats_live(video)
    elif exercise == "pushups":
        generator = count_pushups_live(video)

    # Loop through the generator and yield each frame
    for frame in generator:
        yield frame

interface = gr.Interface(
    fn=process_video_live,
    inputs=[
        gr.Video(format="mp4", sources=["upload"], label="Upload Video", height=600),
        gr.Radio(["squats", "pushups"], value="squats", label="Choose Exercise")
    ],
    outputs=gr.Image(label="Live analysis", height=600),
    flagging_mode="never",
    title="AI-Powered Digital Coach",
    description="<div style='text-align: center;'>Upload a video to count squats or pushups!</div>",
    live=False
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
