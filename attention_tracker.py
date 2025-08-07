import cv2
import numpy as np
import mediapipe as mp

class AttentionTracker:
    def __init__(self):
        # Let's initialize MediaPipe's face mesh tool here.
        # This tool will find all the landmarks on a face.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def analyze_frame(self, frame):
    # MediaPipe works with RGB images, but OpenCV uses BGR.
    # So, first, we need to convert the color space.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Now, we process the frame with our face_mesh tool to find the face.
        results = self.face_mesh.process(frame_rgb)

    # We'll need to check if a face was actually found.
        if results.multi_face_landmarks:
        # We'll just use the first face found.
            face_landmarks = results.multi_face_landmarks[0]

        # Now, let's get the individual attention cues.
        # We'll build these functions next!
            head_pose = self.get_head_pose(face_landmarks, frame.shape)
            gaze = self.get_gaze_direction(face_landmarks, frame)
            blinking = self.is_blinking(face_landmarks)

            attention_score = self.get_attention_score(head_pose, gaze, blinking)
            return {
            "score": attention_score,
            "head_pose": head_pose,
            "gaze": gaze,
            "blinking": blinking
            }
        
    # If no face is found, we'll just return None.
        return None
    
    def get_head_pose(self, face_landmarks, frame_shape):
        # We need the frame's shape to get the camera's focal length.
        # It's a key part of the 3D-to-2D projection.
        size = frame_shape
        focal_length = size[1]
        cam_center = (size[1] / 2, size[0] / 2)
        cam_matrix = np.array(
            [[focal_length, 0, cam_center[0]],
            [0, focal_length, cam_center[1]],
            [0, 0, 1]], dtype=np.double
        )

        # These are the 3D coordinates of a generic face model.
        # We'll match these to the landmarks detected by MediaPipe.
        dist_coeffs = np.zeros((4, 1))
        points_3D = np.array([
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye left corner
            (225.0, 170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),   # Left Mouth corner
            (150.0, -150.0, -125.0)     # Right mouth corner
        ])

        # And these are the specific landmarks from MediaPipe that
        # correspond to the points in our 3D model.
        points_2D = np.array([
            (face_landmarks.landmark[1].x * size[1], face_landmarks.landmark[1].y * size[0]),      # Nose tip
            (face_landmarks.landmark[152].x * size[1], face_landmarks.landmark[152].y * size[0]),   # Chin
            (face_landmarks.landmark[263].x * size[1], face_landmarks.landmark[263].y * size[0]),   # Left eye left corner
            (face_landmarks.landmark[33].x * size[1], face_landmarks.landmark[33].y * size[0]),     # Right eye right corner
            (face_landmarks.landmark[287].x * size[1], face_landmarks.landmark[287].y * size[0]),   # Left Mouth corner
            (face_landmarks.landmark[57].x * size[1], face_landmarks.landmark[57].y * size[0])     # Right mouth corner
        ], dtype=np.double)

        # Now, we use a bit of OpenCV magic to solve for the head's rotation.
        # This function, solvePnP, finds the rotation and translation that
        # best fits the 3D points to the 2D points.
        (success, rot_vec, trans_vec) = cv2.solvePnP(
            points_3D, points_2D, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Finally, we convert the rotation vector into angles we can use.
        (rot_mat, _) = cv2.Rodrigues(rot_vec)
        pose_angles = cv2.RQDecomp3x3(rot_mat)[0]

        return pose_angles
    

    def get_gaze_direction(self, face_landmarks, frame):
    # We need the frame's width and height to convert the landmark
    # coordinates into pixel values.
        frame_height, frame_width, _ = frame.shape

    # These are the specific landmark IDs for the eyes from MediaPipe
    # LEFT EYE:
        left_eye_right_corner = face_landmarks.landmark[33]
        left_eye_left_corner = face_landmarks.landmark[133]
        left_iris = face_landmarks.landmark[473] # Left iris center

    # RIGHT EYE:
        right_eye_right_corner = face_landmarks.landmark[263]
        right_eye_left_corner = face_landmarks.landmark[362]
        right_iris = face_landmarks.landmark[468] # Right iris center

    # --- Gaze calculation for the left eye ---
    # Get the pixel coordinates of the eye corners and iris
        left_iris_x = int(left_iris.x * frame_width)
        left_eye_left_corner_x = int(left_eye_left_corner.x * frame_width)
        left_eye_right_corner_x = int(left_eye_right_corner.x * frame_width)

    # Calculate the horizontal position of the iris relative to the eye corners
        eye_width = left_eye_right_corner_x - left_eye_left_corner_x
    # Avoid division by zero
        if eye_width != 0:
            gaze_ratio_left = (left_iris_x - left_eye_left_corner_x) / eye_width
        else:
            gaze_ratio_left = 0.5 # Assume center if eye width is zero

    # --- Gaze calculation for the right eye (similar logic) ---
        right_iris_x = int(right_iris.x * frame_width)
        right_eye_left_corner_x = int(right_eye_left_corner.x * frame_width)
        right_eye_right_corner_x = int(right_eye_right_corner.x * frame_width)

        eye_width_right = right_eye_right_corner_x - right_eye_left_corner_x
        if eye_width_right != 0:
            gaze_ratio_right = (right_iris_x - right_eye_left_corner_x) / eye_width_right
        else:
            gaze_ratio_right = 0.5

    # Average the ratios from both eyes for a more stable result
        average_gaze_ratio = (gaze_ratio_left + gaze_ratio_right) / 2

    # Use thresholds to determine the final gaze direction
        if average_gaze_ratio < 0.4:
            return "Looking Right" # Remember, it's from the person's perspective
        elif average_gaze_ratio > 0.6:
            return "Looking Left"
        else:
            return "Center"
        
    def is_blinking(self, face_landmarks):
    # We'll use the Eye Aspect Ratio (EAR) to check for blinks.
    # It's a simple ratio of the eye's height to its width.

    # These are the landmark IDs for the vertical and horizontal
    # points of the eyes.
        LEFT_EYE_TOP = face_landmarks.landmark[159]
        LEFT_EYE_BOTTOM = face_landmarks.landmark[145]
        LEFT_EYE_LEFT_CORNER = face_landmarks.landmark[33]
        LEFT_EYE_RIGHT_CORNER = face_landmarks.landmark[133]

        RIGHT_EYE_TOP = face_landmarks.landmark[386]
        RIGHT_EYE_BOTTOM = face_landmarks.landmark[374]
        RIGHT_EYE_LEFT_CORNER = face_landmarks.landmark[362]
        RIGHT_EYE_RIGHT_CORNER = face_landmarks.landmark[263]

    # --- Calculate EAR for the left eye ---
    # Get the vertical distance
        ver_dist_left = self._get_distance(LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
    # Get the horizontal distance
        hor_dist_left = self._get_distance(LEFT_EYE_LEFT_CORNER, LEFT_EYE_RIGHT_CORNER)

    # Calculate the EAR
        ear_left = ver_dist_left / hor_dist_left if hor_dist_left != 0 else 0

    # --- Calculate EAR for the right eye ---
        ver_dist_right = self._get_distance(RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)
        hor_dist_right = self._get_distance(RIGHT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER)
        ear_right = ver_dist_right / hor_dist_right if hor_dist_right != 0 else 0

    # Average the EAR for both eyes
        avg_ear = (ear_left + ear_right) / 2

    # A low EAR value means the eyes are closed.
        BLINK_THRESHOLD = 0.2
        return avg_ear < BLINK_THRESHOLD

# We'll also need a helper function to calculate the distance between two points.
    def _get_distance(self, p1, p2):
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
    
    def get_attention_score(self, head_pose, gaze, is_blinking):
        score = 100
    # We get yaw, pitch, and roll from the head_pose angles
        yaw, pitch, _ = head_pose

    # Penalty for head pose: large penalty if looking away
        if abs(yaw) > 25 or abs(pitch) > 25:
            score -= 50

    # Penalty for gaze: smaller penalty if eyes are averted
        if gaze == "Looking Left" or gaze == "Looking Right":
            score -= 25

    # Penalty for blinking/eyes closed
        if is_blinking:
            score -= 10

    # Ensure the score doesn't go below zero
        return max(0, score)