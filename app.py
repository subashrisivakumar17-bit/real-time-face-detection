import streamlit as st
import cv2

st.title("Real Time Face Detection")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start camera
camera = st.camera_input("Show your face")

if camera:

    # Convert image
    img_bytes = camera.getvalue()

    import numpy as np

    img = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    # Convert gray
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Detect face
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) > 0:
        st.success("Human Face is Detected")
    else:
        st.error("Human Face is Not Detected")

    # Draw rectangle
    for (x,y,w,h) in faces:
        cv2.rectangle(
            img,
            (x,y),
            (x+w,y+h),
            (255,0,0),
            2
        )

    st.image(
        img,
        channels="BGR"
    )