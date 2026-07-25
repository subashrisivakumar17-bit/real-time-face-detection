import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="Real Time Face Detection")

st.title("📷 Real Time Face Detection")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class FaceDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            text = "Human Face is Detected"
            color = (0, 255, 0)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        else:
            text = "Human Face is Not Detected"
            color = (0, 0, 255)

        cv2.putText(
            img,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="face-detection",
    video_processor_factory=FaceDetector,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [
            {
                "urls": ["stun:stun.relay.metered.ca:80"],
            },
            {
                "urls": [
                    "turn:global.relay.metered.ca:80",
                    "turn:global.relay.metered.ca:80?transport=tcp",
                    "turn:global.relay.metered.ca:443",
                    "turns:global.relay.metered.ca:443?transport=tcp",
                ],
                "username": "1b0b30ea045d4345a94c32c3",
                "credential": "trpj8BygLro+bnDt",
            },
        ]
    },
)
