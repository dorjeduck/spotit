
import cv2
import streamlit as st
import mediapipe as mp

from spotit import BlinkDetector,GestureDetector

gesture_model_path = 'models/gesture_recognizer.task'
face_model_path = 'models/face_landmarker_v2_with_blendshapes.task'

# Initialize state for checkboxes and buttons
if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False

    st.session_state.mp_face = mp.solutions.face_mesh
    st.session_state.face_mesh = st.session_state.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    
    st.session_state.mp_hands = mp.solutions.hands
    st.session_state.hands = st.session_state.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    st.session_state.mp_drawing = mp.solutions.drawing_utils

    st.session_state.gesture_detector = GestureDetector(model_path=gesture_model_path)
    st.session_state.blink_detector = BlinkDetector(model_path=face_model_path)

if 'cap' not in st.session_state:
    st.session_state.cap = None

st.title("Spot IT")
# Sidebar for toggling visibility of customization options
st.sidebar.title("Controls")

def toggle_webcam():
    st.session_state.run_webcam = not st.session_state.run_webcam
    if st.session_state.run_webcam:
        st.session_state.cap = cv2.VideoCapture(0)
    else:
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
            cv2.destroyAllWindows()

# Create the buttons and checkboxes
toggle_webcam_button = st.sidebar.button("Stop Webcam" if st.session_state.run_webcam else "Start Webcam", on_click=toggle_webcam)
draw_face_landmarks_checkbox = st.sidebar.checkbox('Draw Face Landmarks')
draw_hand_landmarks_checkbox = st.sidebar.checkbox('Draw Hand Landmarks')
detect_gestures_checkbox = st.sidebar.checkbox('Detect Gestures')
detect_blinks_checkbox = st.sidebar.checkbox('Detect Blinks')

# Define drawing specifications
face_landmark_drawing_spec = st.session_state.mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 0, 255))
face_connection_drawing_spec = st.session_state.mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(255, 255, 0))

hand_landmark_drawing_spec = st.session_state.mp_drawing.DrawingSpec(thickness=4, circle_radius=6, color=(0, 255, 0))
hand_connection_drawing_spec = st.session_state.mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(255, 0, 0))


def process_frame(frame, draw_face_landmarks, draw_hand_landmarks):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if draw_face_landmarks:
        result_face = st.session_state.face_mesh.process(frame_rgb)
        if result_face.multi_face_landmarks:
            for face_landmarks in result_face.multi_face_landmarks:
                st.session_state.mp_drawing.draw_landmarks(
                    image=frame_rgb,  # Draw on the RGB image
                    landmark_list=face_landmarks,
                    connections=st.session_state.mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=face_landmark_drawing_spec,
                    connection_drawing_spec=face_connection_drawing_spec)
                
    if draw_hand_landmarks:
        result_hands = st.session_state.hands.process(frame_rgb)
        if result_hands.multi_hand_landmarks:
            for hand_landmarks in result_hands.multi_hand_landmarks:
                st.session_state.mp_drawing.draw_landmarks(
                    image=frame_rgb,  # Draw on the RGB image
                    landmark_list=hand_landmarks,
                    connections=st.session_state.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_drawing_spec,
                    connection_drawing_spec=hand_connection_drawing_spec)
     
    return frame_rgb


if st.session_state.run_webcam:
    stframe = st.empty()
 
    # Create columns for blink and gesture information
    col1, col2 = st.columns(2)

    # Initialize placeholders in the respective columns
    with col1:
        gesture_placeholder = st.empty()
        
    with col2:
        blink_placeholder = st.empty()
        
    while st.session_state.run_webcam:
        ret, frame = st.session_state.cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame, draw_face_landmarks_checkbox,draw_hand_landmarks_checkbox)
        stframe.image(cv2.flip(processed_frame,1), use_column_width=True)
        #stframe.image(processed_frame, use_column_width=True)

        if detect_blinks_checkbox:
            blinks = st.session_state.blink_detector.detect_blinks(frame)

            def format_value(value):
                return f"{value:.2f}" if value is not None else "N/A"

            blink_info = [
                f"- **Left Eye Blink:** {'**Yes**' if blinks['left_eye_blink_detected'] else 'No'}",
                f"  - Eye Aspect Ratio (EAR): {format_value(blinks['left_eye_ear'])}",
                f"  - Blendshape Score: {format_value(blinks['left_blendshape_blink_score'])}",
                f"\n- **Right Eye Blink:** {'**Yes**' if blinks['right_eye_blink_detected'] else 'No'}",
                f"  - Eye Aspect Ratio (EAR): {format_value(blinks['right_eye_ear'])}",
                f"  - Blendshape Score: {format_value(blinks['right_blendshape_blink_score'])}"
            ]

            blink_placeholder.markdown("<br>".join(blink_info), unsafe_allow_html=True)

        if detect_gestures_checkbox:
            gestures = st.session_state.gesture_detector.detect_gestures(frame)
    
            if gestures[0]['gesture_name'] != '- No Gesture':
                gesture_display = "\n".join([f"- **Gesture:** {gesture['gesture_name']} | **Score:** {gesture['score']:.2f}" for gesture in gestures])
                gesture_placeholder.markdown(gesture_display)
            else:
                gesture_placeholder.markdown("**No Gesture Detected**")
        
    st.session_state.cap.release()
    cv2.destroyAllWindows()
