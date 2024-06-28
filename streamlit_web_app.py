import threading
import cv2
import mediapipe as mp

import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
from time import sleep
from spotit import BlinkDetector, GestureDetector

gesture_model_path = 'models/gesture_recognizer.task'
face_model_path = 'models/face_landmarker_v2_with_blendshapes.task'

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# MediaPipe initialization
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils



# Streamlit UI
st.title("Spot IT")

# Sidebar for toggling visibility of customization options
st.sidebar.title("Controls")
draw_face_landmarks_checkbox = st.sidebar.checkbox('Draw Face Landmarks')
draw_hand_landmarks_checkbox = st.sidebar.checkbox('Draw Hand Landmarks')
detect_gestures_checkbox = st.sidebar.checkbox('Detect Gestures')
detect_blinks_checkbox = st.sidebar.checkbox('Detect Blinks')

# Add an expander for customizable options
with st.sidebar.expander("Customize Detection Options"):
    max_num_faces = st.radio("Max Number of Faces", [1, 2], index=0)
    max_num_hands = st.radio("Max Number of Hands", [1, 2], index=1)
    face_min_detection_confidence = st.slider('Face Min Detection Confidence', 0.0, 1.0, 0.5, step=0.05)
    hand_min_detection_confidence = st.slider('Hand Min Detection Confidence', 0.0, 1.0, 0.5, step=0.05)
    ear_threshold = st.slider('EAR Threshold', 0.0, 1.0, 0.2, step=0.05)
    blendshape_blink_score_threshold = st.slider('Blendshape Blink Score Threshold', 0.0, 1.0, 0.5, step=0.05)

# MediaPipe components with user-defined options
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=max_num_faces,
    min_detection_confidence=face_min_detection_confidence
)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=max_num_hands,
    min_detection_confidence=hand_min_detection_confidence
)
blink_detector = BlinkDetector(
    model_path=face_model_path,
    ear_threshold=ear_threshold,
    blendshape_blink_score_threshold=blendshape_blink_score_threshold
)
gesture_detector = GestureDetector(model_path=gesture_model_path)


# Add an expander for customization of drawing specifications
with st.sidebar.expander("Customize Drawing Specifications"):
    st.subheader("Face Landmark Drawing Specifications")
    face_landmark_thickness = st.slider('Face Landmark Thickness', 1, 10, 2)
    face_landmark_circle_radius = st.slider('Face Landmark Circle Radius', 1, 10, 2)
    face_landmark_color_hex = st.color_picker('Face Landmark Color', '#0000FF')
    face_landmark_color = tuple(int(face_landmark_color_hex[i:i+2], 16) for i in (5, 3, 1))

    st.subheader("Face Connection Drawing Specifications")
    face_connection_thickness = st.slider('Face Connection Thickness', 1, 10, 2)
    face_connection_color_hex = st.color_picker('Face Connection Color', '#FFFF00')
    face_connection_color = tuple(int(face_connection_color_hex[i:i+2], 16) for i in (5, 3, 1))

    st.subheader("Hand Landmark Drawing Specifications")
    hand_landmark_thickness = st.slider('Hand Landmark Thickness', 1, 10, 4)
    hand_landmark_circle_radius = st.slider('Hand Landmark Circle Radius', 1, 10, 6)
    hand_landmark_color_hex = st.color_picker('Hand Landmark Color', '#00FF00')
    hand_landmark_color = tuple(int(hand_landmark_color_hex[i:i+2], 16) for i in (5, 3, 1))

    st.subheader("Hand Connection Drawing Specifications")
    hand_connection_thickness = st.slider('Hand Connection Thickness', 1, 10, 2)
    hand_connection_color_hex = st.color_picker('Hand Connection Color', '#FF0000')
    hand_connection_color = tuple(int(hand_connection_color_hex[i:i+2], 16) for i in (5, 3, 1))

# Define drawing specifications with user inputs
face_landmark_drawing_spec = mp_drawing.DrawingSpec(
    thickness=face_landmark_thickness,
    circle_radius=face_landmark_circle_radius,
    color=face_landmark_color
)
face_connection_drawing_spec = mp_drawing.DrawingSpec(
    thickness=face_connection_thickness,
    color=face_connection_color
)
hand_landmark_drawing_spec = mp_drawing.DrawingSpec(
    thickness=hand_landmark_thickness,
    circle_radius=hand_landmark_circle_radius,
    color=hand_landmark_color
)
hand_connection_drawing_spec = mp_drawing.DrawingSpec(
    thickness=hand_connection_thickness,
    color=hand_connection_color
)


# Initialize threading lock and results container
lock = threading.Lock()
results_container = {"gesture_result": None, "blink_result": None}

def video_frame_callback(frame):
    if frame.format.name == 'yuv420p':
        img = frame.to_ndarray(format="yuv420p")
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_I420)
    else:
        img = frame.to_ndarray(format="bgr24")

    img = cv2.flip(img, 1)

    # Gesture detection
    if detect_gestures_checkbox:
        gestures = gesture_detector.detect_gestures(img, max_num_hands)
       
        with lock:
            if gestures[0]['gesture_name'] != '- No Gesture':
                results_container["gesture_result"] = "\n".join([f"- **Gesture:** {gesture['gesture_name']} | **Score:** {gesture['score']:.2f}" for gesture in gestures])
            else:
                results_container["gesture_result"] = "**No Gesture Detected**"
    # Blink detection
    if detect_blinks_checkbox:
        blinks = blink_detector.detect_blinks(img)
        blink_info = [
            f"- **Left Eye Blink:** {'**Yes**' if blinks['left_eye_blink_detected'] else 'No'}",
            f"  - Eye Aspect Ratio (EAR): {blinks['left_eye_ear']:.2f}" if blinks['left_eye_ear'] is not None else "  - Eye Aspect Ratio (EAR): N/A",
            f"  - Blendshape Score: {blinks['left_blendshape_blink_score']:.2f}" if blinks['left_blendshape_blink_score'] is not None else "  - Blendshape Score: N/A",
            f"\n- **Right Eye Blink:** {'**Yes**' if blinks['right_eye_blink_detected'] else 'No'}",
            f"  - Eye Aspect Ratio (EAR): {blinks['right_eye_ear']:.2f}" if blinks['right_eye_ear'] is not None else "  - Eye Aspect Ratio (EAR): N/A",
            f"  - Blendshape Score: {blinks['right_blendshape_blink_score']:.2f}" if blinks['right_blendshape_blink_score'] is not None else "  - Blendshape Score: N/A"
        ]
        with lock:
            results_container["blink_result"] = "<br>".join(blink_info)
    
   
    # Draw face landmarks
    if draw_face_landmarks_checkbox:
        result_face = face_mesh.process(img)
        if result_face.multi_face_landmarks:
            for face_landmarks in result_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=face_landmark_drawing_spec,
                    connection_drawing_spec=face_connection_drawing_spec
                )

    # Draw hand landmarks
    if draw_hand_landmarks_checkbox:
        result_hands = hands.process(img)
        if result_hands.multi_hand_landmarks:
            for hand_landmarks in result_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_drawing_spec,
                    connection_drawing_spec=hand_connection_drawing_spec
                )

    return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="spotit", 
    video_frame_callback=video_frame_callback, 
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1920},  # Set desired resolution width
            "height": {"ideal": 1080},  # Set desired resolution height
            "frameRate": {"ideal": 30}  # Set desired frame rate
        },
        "audio": False
    },
    video_html_attrs={
        "style": {"width": "100%", "height": "auto"},
        "autoPlay": True,
        "controls": False,
        "muted": True,
    }
)

# Create placeholders for displaying gesture and blink detection results
gesture_placeholder = st.empty()
blink_placeholder = st.empty()

while ctx.state.playing:
    sleep(0.1)
    
    with lock:
        gesture_result = results_container["gesture_result"]
        blink_result = results_container["blink_result"]
    if detect_gestures_checkbox:
        gesture_placeholder.markdown(gesture_result)
    if detect_blinks_checkbox:
        blink_placeholder.markdown(blink_result, unsafe_allow_html=True)
