import streamlit as st
import cv2
import numpy as np

st.title("Real Time Face Detection")

# Load Haar Cascade from project folder
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_cascade.empty():
    st.error("Error: Haar Cascade file not loaded.")
    st.stop()

camera = st.camera_input("Show your face")

if camera is not None:

    img_bytes = camera.getvalue()

    img = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) > 0:
        st.success("Human Face is Detected")
    else:
        st.error("Human Face is Not Detected")

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(img, channels="BGR")