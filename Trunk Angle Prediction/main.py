import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    """Draw a dashed line between two points"""
    dist = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    if dist == 0:
        return

    num_dashes = int(dist / (dash_length * 2))
    for i in range(num_dashes):
        start_ratio = (i * 2 * dash_length) / dist
        end_ratio = ((i * 2 + 1) * dash_length) / dist

        if end_ratio > 1:
            end_ratio = 1

        start_pt = (
            int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio),
            int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio)
        )
        end_pt = (
            int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio),
            int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio)
        )

        cv2.line(img, start_pt, end_pt, color, thickness)


def calculate_trunk_angle(shoulder, hip):
    """
    Calculate trunk angle between shoulder-hip vector and vertical axis
    Points are in (x, y) format where y increases downward
    """
    # Create trunk vector from hip to shoulder
    trunk_vector = np.array([shoulder[0] - hip[0], shoulder[1] - hip[1]])

    # Vertical reference vector (pointing upward in image coordinates)
    vertical_vector = np.array([0, -1])  # y-axis points down, so up is negative

    # Normalize vectors
    trunk_norm = trunk_vector / np.linalg.norm(trunk_vector)
    vertical_norm = vertical_vector / np.linalg.norm(vertical_vector)

    # Calculate angle using dot product
    dot_product = np.dot(trunk_norm, vertical_norm)
    # Clamp to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle in degrees
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def draw_angle_info(image, angle, position):
    """Draw angle information with color coding based on fall risk threshold"""
    # Color coding: Green if safe (≤45°), Red if high risk (>45°)
    color = (0, 255, 0) if angle <= 45 else (0, 0, 255)
    risk_status = "SAFE" if angle <= 45 else "HIGH FALL RISK"

    # Display trunk angle
    cv2.putText(image, f'Trunk Angle: {angle:.1f}°', position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display risk status
    cv2.putText(image, f'Status: {risk_status}',
                (position[0], position[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display threshold reference
    cv2.putText(image, f'Threshold: 45°',
                (position[0], position[1] + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    # Start webcam capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Trunk Angle Detection Started")
    print("- Green: Safe posture (≤45°)")
    print("- Red: High fall risk (>45°)")
    print("Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Process pose detection
        results = pose.process(rgb_frame)

        # Convert back to BGR for OpenCV
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Get shoulder and hip coordinates (using left side for consistency)
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

            # Convert normalized coordinates to pixel coordinates
            shoulder_coords = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            hip_coords = (int(left_hip.x * w), int(left_hip.y * h))

            # Calculate trunk angle
            trunk_angle = calculate_trunk_angle(shoulder_coords, hip_coords)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            # Draw trunk line
            cv2.line(frame, hip_coords, shoulder_coords, (255, 0, 0), 3)

            # Draw vertical reference line (using custom dashed line function)
            ref_start = (hip_coords[0], hip_coords[1])
            ref_end = (hip_coords[0], hip_coords[1] - 100)
            draw_dashed_line(frame, ref_start, ref_end, (255, 255, 255), 2, 8)

            # Display angle information
            draw_angle_info(frame, trunk_angle, (30, 50))

            # Draw key points
            cv2.circle(frame, shoulder_coords, 8, (255, 0, 0), -1)
            cv2.circle(frame, hip_coords, 8, (255, 0, 0), -1)

        else:
            # Display message when no pose is detected
            cv2.putText(frame, 'No pose detected', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Add instructions
        cv2.putText(frame, 'Stand sideways to camera for best results',
                    (30, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow('Trunk Angle Detection - Fall Risk Assessment', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program ended")


if __name__ == "__main__":
    main()
