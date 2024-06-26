import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GestureDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options,num_hands=2)
        self.detector = vision.GestureRecognizer.create_from_options(self.options)
        self.results = []

    def detect_gestures_in_file(self, image_path):
        return self.detect_gestures(cv2.imread(image_path))

    def detect_gestures(self, image):
        self.results = []
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.detector.recognize(mp_image)

        for gesture in detection_result.gestures:
            if gesture:
                top_gesture = gesture[0]
                if top_gesture.category_name != "None":
                    self.results.append({
                        "gesture_name": top_gesture.category_name,
                        "score": top_gesture.score
                    })

        if not self.results:
            self.results.append({
                "gesture_name": "No Gesture",
                "score": 0.0
            })

        return self.results