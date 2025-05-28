import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import math


class FallDetector:
    def __init__(self):
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Motion tracking variables
        self.previous_positions = deque(maxlen=30)  # Store last 30 frames
        self.velocities = deque(maxlen=30)
        self.accelerations = deque(maxlen=30)
        self.previous_time = time.time()

        # Fall detection thresholds based on research findings
        self.fall_velocity_threshold = 80  # m/s [7]
        self.fall_acceleration_threshold = 80.0  # 3g [1]
        self.trunk_angle_threshold = 45  # degrees [10]

        # Status tracking
        self.fall_detected = False
        self.fall_start_time = None

    def calculate_angle(self, a, b, c):
        """
        Calculate angle between three points using the method from [5][6]
        a, b, c are points where b is the vertex
        """
        a = np.array(a)  # First point
        b = np.array(b)  # Mid point (vertex)
        c = np.array(c)  # End point

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def calculate_trunk_angle(self, landmarks):
        """
        Calculate trunk flexion angle based on [10] methodology
        Using shoulder, hip, and a vertical reference
        """
        try:
            # Get shoulder and hip landmarks
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Calculate midpoints
            shoulder_mid = [(left_shoulder[0] + right_shoulder[0]) / 2,
                            (left_shoulder[1] + right_shoulder[1]) / 2]
            hip_mid = [(left_hip[0] + right_hip[0]) / 2,
                       (left_hip[1] + right_hip[1]) / 2]

            # Create vertical reference point
            vertical_ref = [shoulder_mid[0], shoulder_mid[1] + 0.1]

            # Calculate trunk angle relative to vertical
            trunk_angle = self.calculate_angle(vertical_ref, shoulder_mid, hip_mid)

            return trunk_angle, shoulder_mid, hip_mid

        except:
            return None, None, None

    def calculate_vertical_velocity(self, current_position, current_time):
        """
        Calculate vertical velocity by integrating acceleration data [7]
        """
        if len(self.previous_positions) > 0:
            prev_position = self.previous_positions[-1]
            dt = current_time - self.previous_time

            if dt > 0:
                # Calculate velocity (change in position / time)
                velocity_y = (current_position[1] - prev_position[1]) / dt
                self.velocities.append(velocity_y)

                # Calculate acceleration (change in velocity / time)
                if len(self.velocities) > 1:
                    prev_velocity = self.velocities[-2]
                    acceleration_y = (velocity_y - prev_velocity) / dt
                    self.accelerations.append(acceleration_y)

                    return velocity_y, acceleration_y

        return 0, 0

    def detect_fall(self, velocity, acceleration, trunk_angle):
        """
        Detect falls based on motion dynamics and trunk angle [1][7]
        """
        fall_indicators = []

        # Check vertical velocity threshold
        if abs(velocity) > self.fall_velocity_threshold:
            fall_indicators.append("High velocity")

        # Check acceleration threshold (3g from research [1])
        if abs(acceleration) > self.fall_acceleration_threshold:
            fall_indicators.append("High acceleration")

        # Check trunk angle
        if trunk_angle and trunk_angle > self.trunk_angle_threshold:
            fall_indicators.append("Trunk flexion")

        # Fall detection logic
        if len(fall_indicators) >= 2:
            if not self.fall_detected:
                self.fall_detected = True
                self.fall_start_time = time.time()
            return True, fall_indicators
        else:
            # Reset fall detection after 2 seconds of normal activity
            if self.fall_detected and time.time() - self.fall_start_time > 2.0:
                self.fall_detected = False
                self.fall_start_time = None
            return False, []

    def process_frame(self, frame):
        """
        Process each frame for pose detection and fall analysis
        """
        current_time = time.time()

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.pose.process(rgb_frame)

        # Initialize variables
        trunk_angle = None
        velocity = 0
        acceleration = 0
        fall_detected = False
        fall_indicators = []

        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark

            # Calculate trunk angle
            trunk_angle, shoulder_mid, hip_mid = self.calculate_trunk_angle(landmarks)

            if shoulder_mid and hip_mid:
                # Convert normalized coordinates to pixel coordinates
                height, width, _ = frame.shape
                neck_pixel = [int(shoulder_mid[0] * width), int(shoulder_mid[1] * height)]

                # Calculate motion dynamics
                velocity, acceleration = self.calculate_vertical_velocity(neck_pixel, current_time)

                # Store current position
                self.previous_positions.append(neck_pixel)
                self.previous_time = current_time

                # Detect falls
                fall_detected, fall_indicators = self.detect_fall(velocity, acceleration, trunk_angle)

                # Draw trunk line
                hip_pixel = [int(hip_mid[0] * width), int(hip_mid[1] * height)]
                cv2.line(frame, tuple(neck_pixel), tuple(hip_pixel), (0, 255, 0), 3)

        return frame, trunk_angle, velocity, acceleration, fall_detected, fall_indicators

    def run(self):
        """
        Main execution loop for webcam feed processing
        """
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Fall Detection System Started")
        print("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame, trunk_angle, velocity, acceleration, fall_detected, fall_indicators = self.process_frame(
                frame)

            # Display information on frame
            y_offset = 30

            # Trunk angle display
            if trunk_angle is not None:
                cv2.putText(processed_frame, f"Trunk Angle: {trunk_angle:.1f}°",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30

            # Motion dynamics display
            cv2.putText(processed_frame, f"Vertical Velocity: {velocity:.3f} m/s",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

            cv2.putText(processed_frame, f"Vertical Acceleration: {acceleration:.3f} m/s²",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

            # Fall detection status
            if fall_detected:
                cv2.putText(processed_frame, "FALL DETECTED!",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                y_offset += 40

                # Display fall indicators
                for indicator in fall_indicators:
                    cv2.putText(processed_frame, f"- {indicator}",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_offset += 25
            else:
                cv2.putText(processed_frame, "Status: Normal",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display frame
            cv2.imshow('Fall Detection System', processed_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Install required packages (run once)
    # pip install opencv-python mediapipe numpy

    detector = FallDetector()
    detector.run()
