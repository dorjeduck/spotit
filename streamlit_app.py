import cv2
import streamlit as st
import mediapipe as mp

from spotit import BlinkDetector, GestureDetector

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

st.sidebar.text("")
# Create the buttons and checkboxes
toggle_webcam_button = st.sidebar.button("Stop Webcam" if st.session_state.run_webcam else "Start Webcam", on_click=toggle_webcam)

st.sidebar.text("")

draw_face_landmarks_checkbox = st.sidebar.checkbox('Draw Face Landmarks')
draw_hand_landmarks_checkbox = st.sidebar.checkbox('Draw Hand Landmarks')
detect_gestures_checkbox = st.sidebar.checkbox('Detect Gestures')
detect_blinks_checkbox = st.sidebar.checkbox('Detect Blinks')

st.sidebar.markdown("***")

with st.sidebar.expander("Customize Drawing"):

    # Add sliders and color pickers for customization
    st.subheader("Face Drawing")
    face_landmark_thickness = st.slider('Face Landmark Thickness', 1, 10, 2)
    face_landmark_circle_radius = st.slider('Face Landmark Circle Radius', 1, 10, 2)
    face_connection_thickness = st.slider('Face Connection Thickness', 1, 10, 2)
   
    face_landmark_color = st.color_picker('Face Landmark Color', '#0000FF')
    face_landmark_color = tuple(int(face_landmark_color[i:i+2], 16) for i in (1, 3, 5))

    face_connection_color = st.color_picker('Face Connection Color', '#FFFF00')
    face_connection_color = tuple(int(face_connection_color[i:i+2], 16) for i in (1, 3, 5))

    st.subheader("Hand Drawing")
    hand_landmark_thickness = st.slider('Hand Landmark Thickness', 1, 10, 4)
    hand_landmark_circle_radius = st.slider('Hand Landmark Circle Radius', 1, 10, 6)
    hand_connection_thickness = st.slider('Hand Connection Thickness', 1, 10, 2)
    
    hand_landmark_color = st.color_picker('Hand Landmark Color', '#00FF00')
    hand_landmark_color = tuple(int(hand_landmark_color[i:i+2], 16) for i in (1, 3, 5))
    hand_connection_color = st.color_picker('Hand Connection Color', '#FF0000')
    hand_connection_color = tuple(int(hand_connection_color[i:i+2], 16) for i in (1, 3, 5))

    # Define drawing specifications with user inputs
    face_landmark_drawing_spec = st.session_state.mp_drawing.DrawingSpec(
        thickness=face_landmark_thickness,
        circle_radius=face_landmark_circle_radius,
        color=face_landmark_color
    )
    face_connection_drawing_spec = st.session_state.mp_drawing.DrawingSpec(
        thickness=face_connection_thickness,
        color=face_connection_color
    )
    hand_landmark_drawing_spec = st.session_state.mp_drawing.DrawingSpec(
        thickness=hand_landmark_thickness,
        circle_radius=hand_landmark_circle_radius,
        color=hand_landmark_color
    )
    hand_connection_drawing_spec = st.session_state.mp_drawing.DrawingSpec(
        thickness=hand_connection_thickness,
        color=hand_connection_color
    )

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
        processed_frame = process_frame(frame, draw_face_landmarks_checkbox, draw_hand_landmarks_checkbox)
        stframe.image(cv2.flip(processed_frame, 1), use_column_width=True)
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
