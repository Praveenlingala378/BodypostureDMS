"""RTSP-capable posture and drowsiness monitor for Raspberry Pi / desktop."""

import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

# Force RTSP over TCP where OpenCV's FFmpeg backend is available.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    from PIL import ImageTk
    TK_AVAILABLE = True
except ImportError:
    tk = ttk = messagebox = ImageTk = None
    TK_AVAILABLE = False


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
MODEL_DIR = APP_DIR / "models"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"
MODEL_URLS = {
    FACE_MODEL: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    POSE_MODEL: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
FACE_3D_POINTS = [1, 152, 33, 263, 61, 291]
STANDARD_3D = {
    1: (0, 0, 0), 152: (0, -330, -65), 33: (-225, 170, -135),
    263: (225, 170, -135), 61: (-150, -150, -125), 291: (150, -150, -125),
}


def load_config():
    """Load camera settings from the config file beside this script."""
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing {CONFIG_PATH.name}. Add your RTSP URL to it.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{CONFIG_PATH.name} is not valid JSON: {exc}") from exc

    source = str(config.get("rtsp_url", "")).strip()
    if not source.startswith(("rtsp://", "rtsps://")):
        raise RuntimeError('Set "rtsp_url" in config.json to a valid rtsp:// or rtsps:// URL.')
    return source


def ensure_models():
    MODEL_DIR.mkdir(exist_ok=True)
    for model_path, url in MODEL_URLS.items():
        if model_path.exists() and model_path.stat().st_size > 0:
            continue
        temporary_path = model_path.with_suffix(model_path.suffix + ".download")
        print(f"Downloading {model_path.name} (first run only)...")
        try:
            urllib.request.urlretrieve(url, temporary_path)
            temporary_path.replace(model_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def calculate_ear(eye_points, landmarks, img_w, img_h):
    pts = [(int(landmarks[p].x * img_w), int(landmarks[p].y * img_h)) for p in eye_points]
    v1 = math.hypot(pts[1][0] - pts[5][0], pts[1][1] - pts[5][1])
    v2 = math.hypot(pts[2][0] - pts[4][0], pts[2][1] - pts[4][1])
    width = math.hypot(pts[0][0] - pts[3][0], pts[0][1] - pts[3][1])
    return (v1 + v2) / (2.0 * width) if width else 0.0


class SleepDetectionApp:
    def __init__(self, root, video_source):
        self.root = root
        self.video_source = video_source
        self.root.title("Multi-Signal Sleep Detection Dashboard")
        self.cap = self.open_capture()

        self.face_landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        self.ear_start_time = None
        self.head_start_time = None
        self.slouch_start_time = None
        self.baseline_slouch = None
        self.calibration_samples = None
        self.last_capture_time = 0.0
        self.last_timestamp_ms = 0
        self.last_reconnect_attempt = 0.0
        self.frame_count = 0
        self.setup_ui()
        self.update_frame()

    def open_capture(self):
        # Passing CAP_FFMPEG explicitly enables RTSP on common OpenCV builds.
        return cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)

    def setup_ui(self):
        self.video_label = tk.Label(self.root)
        self.video_label.grid(row=0, column=0, rowspan=10, padx=10, pady=10)
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=0, column=1, sticky="nw", padx=10, pady=10)

        self.toggle_ear = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Enable EAR Landmarking", variable=self.toggle_ear,
                       font=("Arial", 10, "bold")).pack(anchor="w")
        self.ear_thresh = tk.Scale(control_frame, from_=0.15, to=0.35, resolution=0.01,
                                   orient="horizontal", label="EAR Threshold (<=)")
        self.ear_thresh.set(0.25)
        self.ear_thresh.pack(anchor="w", fill="x")
        self.ear_time = tk.Scale(control_frame, from_=1, to=5, resolution=0.5,
                                 orient="horizontal", label="EAR Duration (sec)")
        self.ear_time.set(2.0)
        self.ear_time.pack(anchor="w", fill="x")
        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=5)

        self.toggle_hpm = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Enable Head Pose Mapping", variable=self.toggle_hpm,
                       font=("Arial", 10, "bold")).pack(anchor="w")
        self.hpm_thresh = tk.Scale(control_frame, from_=5, to=30, resolution=1,
                                   orient="horizontal", label="Pitch/Roll Degrees (>=)")
        self.hpm_thresh.set(12)
        self.hpm_thresh.pack(anchor="w", fill="x")
        self.hpm_time = tk.Scale(control_frame, from_=1, to=5, resolution=0.5,
                                 orient="horizontal", label="HPM Duration (sec)")
        self.hpm_time.set(2.0)
        self.hpm_time.pack(anchor="w", fill="x")
        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=5)

        self.toggle_slouch = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Enable Body Slouching", variable=self.toggle_slouch,
                       font=("Arial", 10, "bold")).pack(anchor="w")
        self.slouch_time = tk.Scale(control_frame, from_=1, to=5, resolution=0.5,
                                    orient="horizontal", label="Slouch Duration (sec)")
        self.slouch_time.set(2.0)
        self.slouch_time.pack(anchor="w", fill="x")
        self.btn_calibrate = tk.Button(control_frame, text="Calibrate Upright Posture",
                                       command=self.calibrate_posture, bg="lightblue")
        self.btn_calibrate.pack(anchor="w", fill="x", pady=5)

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=10)
        tk.Label(control_frame, text="VERIFICATION STEPPER",
                 font=("Arial", 12, "bold")).pack(anchor="w")
        self.step_ear = tk.Label(control_frame, text="[ ] EAR Monitoring", font=("Arial", 11), fg="gray")
        self.step_ear.pack(anchor="w", pady=2)
        self.step_hpm = tk.Label(control_frame, text="[ ] HPM Monitoring", font=("Arial", 11), fg="gray")
        self.step_hpm.pack(anchor="w", pady=2)
        self.step_slouch = tk.Label(control_frame, text="[ ] Posture Monitoring", font=("Arial", 11), fg="gray")
        self.step_slouch.pack(anchor="w", pady=2)
        self.alert_label = tk.Label(control_frame, text="STATUS: CONNECTING", font=("Arial", 16, "bold"),
                                    fg="white", bg="darkorange", width=22, pady=10)
        self.alert_label.pack(anchor="w", pady=15)

    def calibrate_posture(self):
        self.calibration_samples = []
        self.baseline_slouch = None
        self.btn_calibrate.config(text="Calibrating... face camera upright", bg="khaki")

    def show_incident_report(self, raw_frame, ann_frame, ear_val, pitch_val, roll_val, current_dist, baseline):
        report_win = tk.Toplevel(self.root)
        report_win.title("Sleep Incident Report")
        report_win.geometry("900x600")
        raw_img = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)).resize((400, 300))
        ann_img = Image.fromarray(cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)).resize((400, 300))
        raw_tk, ann_tk = ImageTk.PhotoImage(raw_img), ImageTk.PhotoImage(ann_img)
        report_win.raw_tk, report_win.ann_tk = raw_tk, ann_tk
        tk.Label(report_win, text="Actual Image (Raw Camera Feed)", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=10)
        tk.Label(report_win, text="Annotated Image (System View)", font=("Arial", 12, "bold")).grid(row=0, column=1, pady=10)
        tk.Label(report_win, image=raw_tk).grid(row=1, column=0, padx=10)
        tk.Label(report_win, image=ann_tk).grid(row=1, column=1, padx=10)
        slouch_pct = (current_dist / baseline) * 100 if baseline and baseline > 0 else 100
        reason = ("--- INCIDENT LOG ---\n"
                  f"1. Eye Aspect Ratio: {ear_val:.2f} for over {self.ear_time.get()}s.\n"
                  f"2. Head Pose: Pitch {int(pitch_val)}° / Roll {int(roll_val)}° for over {self.hpm_time.get()}s.\n"
                  f"3. Body Posture: {slouch_pct:.1f}% of calibrated upright distance for over {self.slouch_time.get()}s.\n"
                  "CONCLUSION: Enabled multi-signal checks confirmed a drowsiness event.")
        tk.Label(report_win, text=reason, font=("Consolas", 11, "bold"), justify="left",
                 fg="darkred", bg="#ffe6e6", padx=10, pady=10).grid(row=2, column=0, columnspan=2, pady=20, sticky="ew")
        tk.Button(report_win, text="Dismiss & Resume Monitoring", command=report_win.destroy,
                  font=("Arial", 12), bg="lightgray", height=2).grid(row=3, column=0, columnspan=2, pady=10)

    def reconnect_if_needed(self):
        if self.cap.isOpened():
            return
        now = time.monotonic()
        if now - self.last_reconnect_attempt >= 2.0:
            self.last_reconnect_attempt = now
            self.cap.release()
            self.cap = self.open_capture()
        self.alert_label.config(text="STATUS: RTSP RECONNECTING", fg="white", bg="darkorange")

    def update_frame(self):
        if not self.cap.isOpened():
            self.reconnect_if_needed()
            self.root.after(500, self.update_frame)
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.cap.release()
            self.reconnect_if_needed()
            self.root.after(500, self.update_frame)
            return

        # Limit inference resolution to reduce CPU and memory load on Pi 4B.
        height, width = frame.shape[:2]
        if width > 640:
            new_height = max(1, round(height * 640 / width))
            frame = cv2.resize(frame, (640, new_height), interpolation=cv2.INTER_AREA)
        raw_frame_copy = frame.copy()
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.frame_count += 1
        timestamp_ms = max(int(time.monotonic() * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms

        face_results = None
        if self.toggle_ear.get() or self.toggle_hpm.get():
            face_results = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_results = None
        if self.toggle_slouch.get():
            pose_results = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        ear_triggered = hpm_triggered = slouch_triggered = False
        current_ear = current_pitch = current_roll = current_dist = 0.0
        ear_elapsed = hpm_elapsed = slouch_elapsed = 0.0
        face_found = bool(face_results and face_results.face_landmarks)
        pose_found = bool(pose_results and pose_results.pose_landmarks)

        if face_found:
            landmarks = face_results.face_landmarks[0]
            if self.toggle_ear.get():
                current_ear = (calculate_ear(LEFT_EYE, landmarks, w, h) +
                               calculate_ear(RIGHT_EYE, landmarks, w, h)) / 2.0
                for point_idx in LEFT_EYE + RIGHT_EYE:
                    point = (int(landmarks[point_idx].x * w), int(landmarks[point_idx].y * h))
                    cv2.circle(frame, point, 2, (0, 255, 0), -1)
                cv2.putText(frame, f"EAR: {current_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if current_ear <= self.ear_thresh.get():
                    if self.ear_start_time is None:
                        self.ear_start_time = time.monotonic()
                    ear_elapsed = time.monotonic() - self.ear_start_time
                    ear_triggered = ear_elapsed >= self.ear_time.get()
                else:
                    self.ear_start_time = None
            else:
                self.ear_start_time = None

            if self.toggle_hpm.get():
                image_points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                                         for i in FACE_3D_POINTS], dtype=np.float64)
                model_points = np.array([STANDARD_3D[i] for i in FACE_3D_POINTS], dtype=np.float64)
                camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
                success, rotation, translation = cv2.solvePnP(
                    model_points, image_points, camera_matrix, np.zeros((4, 1), dtype=np.float64))
                if success:
                    rotation_matrix, _ = cv2.Rodrigues(rotation)
                    angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
                    current_pitch, current_roll = float(angles[0] * 360), float(angles[2] * 360)
                    cv2.putText(frame, f"Pitch: {int(current_pitch)} Roll: {int(current_roll)}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    if abs(current_pitch) >= self.hpm_thresh.get() or abs(current_roll) >= self.hpm_thresh.get():
                        if self.head_start_time is None:
                            self.head_start_time = time.monotonic()
                        hpm_elapsed = time.monotonic() - self.head_start_time
                        hpm_triggered = hpm_elapsed >= self.hpm_time.get()
                    else:
                        self.head_start_time = None
                else:
                    self.head_start_time = None
            else:
                self.head_start_time = None
        else:
            # A gap in face tracking breaks the continuous-duration condition.
            self.ear_start_time = self.head_start_time = None

        if pose_found and self.toggle_slouch.get():
            pose = pose_results.pose_landmarks[0]
            nose = (int(pose[0].x * w), int(pose[0].y * h))
            shoulders = (int((pose[11].x + pose[12].x) / 2 * w),
                         int((pose[11].y + pose[12].y) / 2 * h))
            current_dist = shoulders[1] - nose[1]
            cv2.line(frame, nose, shoulders, (255, 0, 255), 2)
            cv2.circle(frame, shoulders, 5, (255, 0, 255), -1)
            if self.calibration_samples is not None and len(self.calibration_samples) < 20:
                self.calibration_samples.append(current_dist)
                if len(self.calibration_samples) == 20:
                    self.baseline_slouch = float(np.median(self.calibration_samples))
                    self.calibration_samples = None
                    self.btn_calibrate.config(text="Upright Posture Calibrated", bg="lightgreen")
            if self.baseline_slouch and self.baseline_slouch > 0:
                if current_dist < self.baseline_slouch * 0.8:
                    cv2.putText(frame, "SLOUCH DETECTED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if self.slouch_start_time is None:
                        self.slouch_start_time = time.monotonic()
                    slouch_elapsed = time.monotonic() - self.slouch_start_time
                    slouch_triggered = slouch_elapsed >= self.slouch_time.get()
                else:
                    self.slouch_start_time = None
            else:
                self.slouch_start_time = None
        else:
            self.slouch_start_time = None

        criteria = [
            (self.toggle_ear.get(), ear_triggered, self.ear_start_time, ear_elapsed,
             self.step_ear, "EAR", self.ear_time.get()),
            (self.toggle_hpm.get(), hpm_triggered, self.head_start_time, hpm_elapsed,
             self.step_hpm, "HPM", self.hpm_time.get()),
            (self.toggle_slouch.get(), slouch_triggered, self.slouch_start_time, slouch_elapsed,
             self.step_slouch, "Posture", self.slouch_time.get()),
        ]
        for enabled, triggered, start_time, elapsed, label, name, duration in criteria:
            if not enabled:
                label.config(text=f"✓ {name} Disabled", fg="green")
            elif triggered:
                label.config(text=f"✓ {name} Met ({duration:.1f}s)", fg="green")
            elif start_time is not None:
                label.config(text=f"[ ] {name} Triggering... ({elapsed:.1f}s / {duration:g}s)", fg="orange")
            elif name == "Posture" and not self.baseline_slouch:
                label.config(text="[ ] Calibrate upright posture", fg="orange")
            elif name in ("EAR", "HPM") and not face_found:
                label.config(text=f"[ ] {name}: waiting for face", fg="gray")
            elif name == "Posture" and not pose_found:
                label.config(text="[ ] Posture: waiting for person", fg="gray")
            else:
                label.config(text=f"[ ] {name} Monitoring", fg="gray")

        enabled_criteria = [enabled for enabled, *_ in criteria]
        detection_confirmed = any(enabled_criteria) and all((not enabled) or met for enabled, met, *_ in criteria)
        if detection_confirmed:
            self.alert_label.config(text="SLEEPINESS DETECTED", fg="white", bg="red")
            if time.monotonic() - self.last_capture_time > 10:
                self.show_incident_report(raw_frame_copy, frame.copy(), current_ear,
                                          current_pitch, current_roll, current_dist, self.baseline_slouch)
                self.last_capture_time = time.monotonic()
                self.ear_start_time = self.head_start_time = self.slouch_start_time = None
        else:
            self.alert_label.config(text="STATUS: AWAKE", fg="green", bg="black")

        display = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(display)
        self.video_label.imgtk = photo
        self.video_label.configure(image=photo)
        # 30ms caps polling at about 33fps; inference resolution is capped above.
        self.root.after(30, self.update_frame)

    def on_closing(self):
        self.cap.release()
        self.face_landmarker.close()
        self.pose_landmarker.close()
        self.root.destroy()


def run_headless(video_source):
    """Monitor the RTSP stream from SSH without opening a window."""
    face_landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    print("Headless monitoring started. Press Ctrl+C to stop.", flush=True)
    print("Posture baseline will be calibrated from the first 20 detected frames; sit upright.", flush=True)

    ear_start = head_start = slouch_start = None
    last_timestamp_ms = 0
    last_reconnect = 0.0
    calibration_samples = []
    baseline = None
    alert_active = False
    last_status = None

    try:
        while True:
            if not cap.isOpened():
                now = time.monotonic()
                if now - last_reconnect >= 2.0:
                    cap.release()
                    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
                    last_reconnect = now
                status = "Waiting for RTSP stream"
                if status != last_status:
                    print(f"[INFO] {status}", flush=True)
                    last_status = status
                time.sleep(0.25)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                ear_start = head_start = slouch_start = None
                status = "RTSP stream lost; reconnecting"
                if status != last_status:
                    print(f"[WARN] {status}", flush=True)
                    last_status = status
                time.sleep(0.5)
                continue

            height, width = frame.shape[:2]
            if width > 640:
                frame = cv2.resize(frame, (640, max(1, round(height * 640 / width))),
                                   interpolation=cv2.INTER_AREA)
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = max(int(time.monotonic() * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            now = time.monotonic()
            ear_met = head_met = slouch_met = False
            ear_value = pitch = roll = 0.0

            if face_result.face_landmarks:
                face = face_result.face_landmarks[0]
                ear_value = (calculate_ear(LEFT_EYE, face, width, height) +
                             calculate_ear(RIGHT_EYE, face, width, height)) / 2.0
                if ear_value <= 0.25:
                    if ear_start is None:
                        ear_start = now
                    ear_met = now - ear_start >= 2.0
                else:
                    ear_start = None

                points_2d = np.array([[int(face[i].x * width), int(face[i].y * height)]
                                      for i in FACE_3D_POINTS], dtype=np.float64)
                points_3d = np.array([STANDARD_3D[i] for i in FACE_3D_POINTS], dtype=np.float64)
                camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
                                  dtype=np.float64)
                success, rotation, _ = cv2.solvePnP(
                    points_3d, points_2d, camera, np.zeros((4, 1), dtype=np.float64))
                if success:
                    rotation_matrix, _ = cv2.Rodrigues(rotation)
                    angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
                    pitch, roll = float(angles[0] * 360), float(angles[2] * 360)
                    if abs(pitch) >= 12 or abs(roll) >= 12:
                        if head_start is None:
                            head_start = now
                        head_met = now - head_start >= 2.0
                    else:
                        head_start = None
                else:
                    head_start = None
            else:
                ear_start = head_start = None

            if pose_result.pose_landmarks:
                pose = pose_result.pose_landmarks[0]
                nose_y = pose[0].y * height
                shoulder_y = (pose[11].y + pose[12].y) * height / 2.0
                distance = shoulder_y - nose_y
                if baseline is None:
                    if distance > 0:
                        calibration_samples.append(distance)
                    if len(calibration_samples) >= 20:
                        baseline = float(np.median(calibration_samples))
                        print(f"[INFO] Upright posture baseline calibrated: {baseline:.1f}px", flush=True)
                elif distance < baseline * 0.8:
                    if slouch_start is None:
                        slouch_start = now
                    slouch_met = now - slouch_start >= 2.0
                else:
                    slouch_start = None
            else:
                slouch_start = None

            # Preserve the dashboard's multi-signal rule: all enabled checks must meet.
            is_alert = ear_met and head_met and slouch_met
            if is_alert and not alert_active:
                print(f"[ALERT] Sleepiness detected | EAR={ear_value:.2f}, "
                      f"pitch={pitch:.1f}°, roll={roll:.1f}°", flush=True)
            elif not is_alert and alert_active:
                print("[INFO] Alert cleared; monitoring resumed.", flush=True)
            alert_active = is_alert

            status = "Monitoring RTSP stream" if baseline is not None else "Calibrating posture baseline"
            if status != last_status:
                print(f"[INFO] {status}", flush=True)
                last_status = status
    except KeyboardInterrupt:
        print("\nHeadless monitoring stopped.", flush=True)
    finally:
        cap.release()
        face_landmarker.close()
        pose_landmarker.close()


if __name__ == "__main__":
    try:
        source = load_config()
        ensure_models()
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not TK_AVAILABLE or (sys.platform.startswith("linux") and not has_display):
            run_headless(source)
        else:
            root = tk.Tk()
            app = SleepDetectionApp(root, source)
            root.protocol("WM_DELETE_WINDOW", app.on_closing)
            root.mainloop()
    except Exception as error:
        print(f"Startup failed: {error}")
        if TK_AVAILABLE and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            try:
                messagebox.showerror("Startup failed", str(error))
            except tk.TclError:
                pass
