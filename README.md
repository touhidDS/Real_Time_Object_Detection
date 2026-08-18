# Real-Time Object Detection and Tracking

YOLOv8 + DeepSORT project — detects objects in a video/webcam feed and gives each one a
persistent ID as it moves across frames, instead of just re-detecting it fresh every frame.

Two ways to run it:
- `local_demo.py` — plain script, opens an OpenCV window on your machine.
- `app.py` — Streamlit web app with live webcam and video upload modes.

## How it works

Detection and tracking are two separate problems. YOLOv8 finds objects in a single frame
(box + class + confidence), but it has no idea that the person in frame 10 is the same person as
in frame 9 — every frame is independent. DeepSORT is what adds continuity: it keeps a Kalman
filter estimate of where each tracked object should be next, and pairs that with an appearance
embedding (MobileNet) so it can still match objects after a brief occlusion or a jump in position.
When a detection matches an existing track, it keeps the same ID. If nothing matches for
`max_age` frames, the track gets dropped.

Pipeline per frame:
1. Read frame → run YOLOv8 → get boxes/classes/confidences
2. Filter by confidence threshold, feed the rest to the DeepSORT tracker
3. Tracker matches detections to existing tracks (or creates new ones)
4. Draw boxes + class + track ID back on the frame

## Tech stack

- `ultralytics` for YOLOv8
- `deep-sort-realtime` for tracking
- OpenCV for video I/O and drawing
- Streamlit + `streamlit-webrtc` for the web app
- Twilio's TURN service so the webcam mode can work once deployed (see below)

## Running locally

```
git clone https://github.com/touhidDS/Real_Time_Object_Detection.git
cd Real_Time_Object_Detection
python -m venv venv
venv\Scripts\activate   # or source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

YOLOv8n weights (~6MB) download automatically on first run.

For `local_demo.py`, swap in the GUI build of OpenCV first since `requirements.txt` has the
headless version (needed for cloud deployment):
```
pip install opencv-python
python local_demo.py                       # webcam
python local_demo.py --source video.mp4     # video file
python local_demo.py --save output.mp4       # also save the annotated output
```
Press `q` to quit. Run `python local_demo.py -h` for the full list of flags (confidence
threshold, tracker params, etc).

For the web app:
```
streamlit run app.py
```

## Deploying (Streamlit Community Cloud)

Push to GitHub, then point Streamlit Cloud at the repo with `app.py` as the entry point.
`requirements.txt` and `packages.txt` (system libs OpenCV needs on Linux) get installed
automatically.

The webcam mode needs one extra step once deployed: Streamlit Cloud's network doesn't allow
direct peer-to-peer WebRTC, so it needs a TURN relay or it just won't connect. I used Twilio's
free TURN service for this:

1. Make a free Twilio account, grab the Account SID and Auth Token from the console.
2. In the deployed app's settings → Secrets, add:
   ```
   TWILIO_ACCOUNT_SID = "..."
   TWILIO_AUTH_TOKEN = "..."
   ```
3. For local testing, copy `secrets.toml.example` to `.streamlit/secrets.toml` with the same values.

Without these it falls back to public STUN only, which is fine on localhost but generally won't
connect on the deployed app. Video upload mode doesn't need any of this, it works out of the box.

## Notes / limitations

- ID switches can still happen — if a detection is missed for a few frames (motion blur,
  occlusion, low confidence) the tracker sometimes can't match it back and gives it a new ID.
  Tuning `max_age` / `n_init` / `max_cosine_distance` helps but doesn't eliminate it.
- Free tier hosting is CPU-only, so expect ~3-8 FPS with the nano model, not real-time-smooth.
- Twilio's TURN service is usage-billed — the free trial covers demo/testing but not sustained traffic.

Possible next steps: fine-tune on a custom dataset, add zone/line-crossing counting, log
trajectories for analysis, class filtering in the UI.
