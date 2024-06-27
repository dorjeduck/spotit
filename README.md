# SpotIT

SpotIT is a real-time computer vision application that demonstrates the use of [MediaPipe](https://github.com/google-ai-edge/mediapipe) and [OpenCV](https://opencv.org/) for gesture recognition and blink detection. The web interface is built with [Streamlit](https://streamlit.io/).

## Features

- **Real-time Gesture Recognition:** Detects various hand gestures using MediaPipe.
- **Blink Detection:** Detects eye blinks by calculating the Eye Aspect Ratios (EAR) and blendshape scores.
- **Face and Hand Landmark Visualization:** Displays the face and hand landmarks and mesh as recognized by MediaPipe.

![SpotIT](imgs/demo.png)

## Customization

- **Webcam Control:** Start or stop the webcam feed.
- **Display Options:**
  - Toggle the display of face landmarks.
  - Toggle the display of hand landmarks.
- **Gesture and Blink Detection:**
  - Enable or disable gesture detection.
  - Enable or disable blink detection.
- **Detection Options:**
  - **Max Number of Faces:** Choose to detect 1 or 2 faces.
  - **Max Number of Hands:** Choose to detect 1 or 2 hands.
  - **Face Min Detection Confidence:** Adjust the minimum confidenc threshold for face detection (0.0 to 1.0).
  - **Hand Min Detection Confidence:** Adjust the minimum confidence threshold for hand detection (0.0 to 1.0).
  - **EAR Threshold:** Set the Eye Aspect Ratio (EAR) threshold for blink detection (0.0 to 1.0). A value below this threshold indicates a blink.
  - **Blendshape Blink Score Threshold:** Set the blendshape blink score threshold for blink detection (0.0 to 1.0). A value above this threshold indicates a blink.
- **Drawing Specifications:**
  - Customize the thickness, circle radius, and color of face landmarks.
  - Customize the thickness and color of face connections.
  - Customize the thickness, circle radius, and color of hand landmarks.
  - Customize the thickness and color of hand connections.
  
## Installation

Ensure you have Python >= 3.9 installed. To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To start the application, run:

```bash
streamlit run streamlit_app.py
```

This will open the application in your default web browser.

## Changelog

- 2024-06-27
  - Added drawing options
  - Added detection options
- 2024-06-26
  - Initial commit

## License

This project is licensed under the MIT License.
