import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist
from enum import Enum

class DetectionCriteria(Enum):
    EAR_AND_BLENDSHAPE = "ear_and_blendshape"
    EAR_ONLY = "ear_only"
    BLENDSHAPE_ONLY = "blendshape_only"
    EITHER = "either"

class BlinkDetector:
    def __init__(self, model_path, ear_threshold=0.2, blendshape_blink_score_threshold=0.5, detection_criteria=DetectionCriteria.EAR_AND_BLENDSHAPE):
        self.model_path = model_path
        self.ear_threshold = ear_threshold
        self.blendshape_blink_score_threshold = blendshape_blink_score_threshold
        self.detection_criteria = detection_criteria
        self.base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(self.options)
        self.results = {
            "left_eye_ear": None,
            "right_eye_ear": None,
            "left_blendshape_blink_score": None,
            "right_blendshape_blink_score": None,
            "left_eye_blink_detected": False,
            "right_eye_blink_detected": False
        }

    def calculate_ear(self, eye_landmarks):
        # Calculate the distances between the vertical landmarks
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        # Calculate the distance between the horizontal landmarks
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        # Compute the EAR
        ear = (A + B) / (2.0 * C)
        return ear

    def extract_eye_landmarks(self, landmarks, eye_indices):
        return [(landmarks[idx].x, landmarks[idx].y) for idx in eye_indices]

    def detect_blinks_in_file(self,image_path):     
        return self.detect_blinks(cv2.imread(image_path))

    def detect_blinks(self, image):
        self.results = {
            "left_eye_ear": None,
            "right_eye_ear": None,
            "left_blendshape_blink_score": None,
            "right_blendshape_blink_score": None,
            "left_eye_blink_detected": False,
            "right_eye_blink_detected": False
        }
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detection_result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))

        face_landmarks_list = detection_result.face_landmarks
        blendshapes_list = detection_result.face_blendshapes

        # Indices for eye landmarks
        left_eye_indices = [362, 385, 387, 263, 373, 380]  # Example indices, may vary based on your model
        right_eye_indices = [33, 160, 158, 133, 153, 144]  # Example indices, may vary based on your model
        
        if face_landmarks_list:
            for face_landmarks in face_landmarks_list:
                landmarks = [landmark for landmark in face_landmarks]

                if self.detection_criteria in [DetectionCriteria.EAR_AND_BLENDSHAPE, DetectionCriteria.EAR_ONLY, DetectionCriteria.EITHER]:
                    # Calculate EAR for the left eye
                    left_eye_landmarks = self.extract_eye_landmarks(landmarks, left_eye_indices)
                    left_ear = self.calculate_ear(left_eye_landmarks)
                    self.results["left_eye_ear"] = left_ear

                    # Calculate EAR for the right eye
                    right_eye_landmarks = self.extract_eye_landmarks(landmarks, right_eye_indices)
                    right_ear = self.calculate_ear(right_eye_landmarks)
                    self.results["right_eye_ear"] = right_ear

                    # EAR-based blink detection
                    left_ear_detected = left_ear < self.ear_threshold
                    right_ear_detected = right_ear < self.ear_threshold

                if self.detection_criteria in [DetectionCriteria.EAR_AND_BLENDSHAPE, DetectionCriteria.BLENDSHAPE_ONLY, DetectionCriteria.EITHER]:
                    # Blendshape-based blink detection
                    blendshapes = blendshapes_list[0]
                    left_blendshape_blink_score = next((category.score for category in blendshapes if category.category_name == 'eyeBlinkLeft'), 0.0)
                    right_blendshape_blink_score = next((category.score for category in blendshapes if category.category_name == 'eyeBlinkRight'), 0.0)
                    self.results["left_blendshape_blink_score"] = left_blendshape_blink_score
                    self.results["right_blendshape_blink_score"] = right_blendshape_blink_score

                # Detection based on criteria
                if self.detection_criteria == DetectionCriteria.EAR_AND_BLENDSHAPE:
                    self.results["left_eye_blink_detected"] = left_ear_detected and (left_blendshape_blink_score > self.blendshape_blink_score_threshold)
                    self.results["right_eye_blink_detected"] = right_ear_detected and (right_blendshape_blink_score > self.blendshape_blink_score_threshold)
                elif self.detection_criteria == DetectionCriteria.EAR_ONLY:
                    self.results["left_eye_blink_detected"] = left_ear_detected
                    self.results["right_eye_blink_detected"] = right_ear_detected
                elif self.detection_criteria == DetectionCriteria.BLENDSHAPE_ONLY:
                    self.results["left_eye_blink_detected"] = left_blendshape_blink_score > self.blendshape_blink_score_threshold
                    self.results["right_eye_blink_detected"] = right_blendshape_blink_score > self.blendshape_blink_score_threshold
                elif self.detection_criteria == DetectionCriteria.EITHER:
                    self.results["left_eye_blink_detected"] = left_ear_detected or (left_blendshape_blink_score > self.blendshape_blink_score_threshold)
                    self.results["right_eye_blink_detected"] = right_ear_detected or (right_blendshape_blink_score > self.blendshape_blink_score_threshold)

        return self.results
