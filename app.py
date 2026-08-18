import av
import cv2
import time
import logging
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

logger = logging.getLogger(__name__)


def get_ice_servers():
    # Streamlit Cloud blocks direct peer-to-peer WebRTC, so without a TURN
    # server the webcam mode won't connect once deployed. Using Twilio's
    # free TURN service here, falls back to STUN only if no creds are set
    # (works on localhost, not on the deployed app).
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
    except (KeyError, FileNotFoundError):
        logger.warning("No Twilio credentials in st.secrets, using public STUN only.")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    from twilio.rest import Client
    client = Client(account_sid, auth_token)
    token = client.tokens.create()
    return token.ice_servers


st.set_page_config(page_title="Object Detection & Tracking", layout="wide")
st.title("Real-Time Object Detection & Tracking")
st.caption("YOLOv8 + DeepSORT — detects objects and tracks them with persistent IDs")

st.sidebar.header("Settings")
model_name = st.sidebar.selectbox(
    "Model (bigger = more accurate, slower)",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0,
)
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
max_age = st.sidebar.slider("Track max age (frames to keep a lost track)", 5, 60, 30, 5)
n_init = st.sidebar.slider("Frames to confirm a new track (n_init)", 1, 5, 3, 1)
max_cosine_distance = st.sidebar.slider(
    "Appearance match leniency (higher = fewer ID switches, more risk of confusion)",
    0.1, 0.6, 0.3, 0.05
)
mode = st.sidebar.radio("Mode", ["Live Webcam", "Upload Video"])


@st.cache_resource
def load_model(name):
    return YOLO(name)


model = load_model(model_name)

# fixed seed so a class gets the same box color every run
np.random.seed(42)
COLORS = {i: tuple(int(c) for c in np.random.randint(0, 255, 3)) for i in range(len(model.names))}


def detect_and_track(frame, model, tracker, conf_threshold):
    results = model(frame, verbose=False, conf=conf_threshold)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, model.names[cls]))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        cls_name = track.get_det_class() or "object"
        l, t, r, b = map(int, track.to_ltrb())
        color = COLORS.get(hash(cls_name) % len(COLORS), (0, 255, 0))

        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        label = f"{cls_name} | ID {track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (l, t - th - 8), (l + tw + 4, t), color, -1)
        cv2.putText(frame, label, (l + 2, t - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return frame


if mode == "Live Webcam":
    st.info("Click **Start** below and allow camera access. Video is processed on the server "
            "and streamed back, it isn't stored anywhere.")

    class Processor(VideoProcessorBase):
        def __init__(self):
            self.tracker = DeepSort(max_age=max_age, n_init=n_init, max_cosine_distance=max_cosine_distance, embedder="mobilenet")

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = detect_and_track(img, model, self.tracker, conf_threshold)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": get_ice_servers()})

    webrtc_streamer(
        key="object-tracking",
        video_processor_factory=Processor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

else:
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        tracker = DeepSort(max_age=max_age, n_init=n_init, max_cosine_distance=max_cosine_distance, embedder="mobilenet")
        progress = st.progress(0, text="Processing video...")
        frame_placeholder = st.empty()

        frame_idx = 0
        start = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect_and_track(frame, model, tracker, conf_threshold)
            writer.write(frame)

            frame_idx += 1
            if frame_idx % 5 == 0:  # updating the preview every frame kills the UI, so skip some
                frame_placeholder.image(frame, channels="BGR", caption=f"Frame {frame_idx}/{total_frames}")
                if total_frames > 0:
                    progress.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        writer.release()
        elapsed = time.time() - start
        progress.progress(1.0, text=f"Done in {elapsed:.1f}s")

        st.success("Processing complete!")
        st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download processed video", f, file_name="tracked_output.mp4")
    else:
        st.write("👆 Upload a video file to run detection + tracking on it.")
