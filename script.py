import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import math

# MediaPipe Initializations
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
FACE_3D_POINTS = [1, 152, 33, 263, 61, 291] 

def calculate_ear(eye_points, landmarks, img_w, img_h):
    pts = [(int(landmarks.landmark[p].x * img_w), int(landmarks.landmark[p].y * img_h)) for p in eye_points]
    v1 = math.hypot(pts[1][0] - pts[5][0], pts[1][1] - pts[5][1])
    v2 = math.hypot(pts[2][0] - pts[4][0], pts[2][1] - pts[4][1])
    h = math.hypot(pts[0][0] - pts[3][0], pts[0][1] - pts[3][1])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0

class SleepDetectionApp:
    def __init__(self, root, video_source=0):
        self.root = root
        self.root.title("Multi-Signal Sleep Detection Dashboard")
        self.cap = cv2.VideoCapture(video_source)
        
        self.ear_start_time = None
        self.head_start_time = None
        self.slouch_start_time = None
        self.baseline_slouch = None
        self.last_capture_time = 0 
        
        self.setup_ui()
        self.update_frame()

    def setup_ui(self):
        self.video_label = tk.Label(self.root)
        self.video_label.grid(row=0, column=0, rowspan=10, padx=10, pady=10)

        control_frame = tk.Frame(self.root)
        control_frame.grid(row=0, column=1, sticky="nw", padx=10, pady=10)

        self.toggle_ear = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Enable EAR Landmarking", variable=self.toggle_ear, font=("Arial", 10, "bold")).pack(anchor="w")
        self.ear_thresh = tk.Scale(control_frame, from_=0.15, to=0.35, resolution=0.01, orient="horizontal", label="EAR Threshold (<=)")
        self.ear_thresh.set(0.25)
        self.ear_thresh.pack(anchor="w", fill="x")
        self.ear_time = tk.Scale(control_frame, from_=1, to=5, resolution=0.5, orient="horizontal", label="EAR Duration (sec)")
        self.ear_time.set(2.0)
        self.ear_time.pack(anchor="w", fill="x")

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=5)

        self.toggle_hpm = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Enable Head Pose Mapping", variable=self.toggle_hpm, font=("Arial", 10, "bold")).pack(anchor="w")
        self.hpm_thresh = tk.Scale(control_frame, from_=5, to=30, resolution=1, orient="horizontal", label="Pitch/Roll Degrees (>=)")
        self.hpm_thresh.set(12)
        self.hpm_thresh.pack(anchor="w", fill="x")
        self.hpm_time = tk.Scale(control_frame, from_=1, to=5, resolution=0.5, orient="horizontal", label="HPM Duration (sec)")
        self.hpm_time.set(2.0)
        self.hpm_time.pack(anchor="w", fill="x")

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=5)

        self.toggle_slouch = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Enable Body Slouching", variable=self.toggle_slouch, font=("Arial", 10, "bold")).pack(anchor="w")
        self.slouch_time = tk.Scale(control_frame, from_=1, to=5, resolution=0.5, orient="horizontal", label="Slouch Duration (sec)")
        self.slouch_time.set(2.0)
        self.slouch_time.pack(anchor="w", fill="x")
        self.btn_calibrate = tk.Button(control_frame, text="Calibrate Upright Posture", command=self.calibrate_posture, bg="lightblue")
        self.btn_calibrate.pack(anchor="w", fill="x", pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        tk.Label(control_frame, text="VERIFICATION STEPPER", font=("Arial", 12, "bold")).pack(anchor="w")
        self.step_ear = tk.Label(control_frame, text="[ ] EAR Monitoring", font=("Arial", 11), fg="gray")
        self.step_ear.pack(anchor="w", pady=2)
        self.step_hpm = tk.Label(control_frame, text="[ ] HPM Monitoring", font=("Arial", 11), fg="gray")
        self.step_hpm.pack(anchor="w", pady=2)
        self.step_slouch = tk.Label(control_frame, text="[ ] Posture Monitoring", font=("Arial", 11), fg="gray")
        self.step_slouch.pack(anchor="w", pady=2)

        self.alert_label = tk.Label(control_frame, text="STATUS: ACTIVE", font=("Arial", 16, "bold"), fg="green", bg="black", width=20, pady=10)
        self.alert_label.pack(anchor="w", pady=15)

    def calibrate_posture(self):
        self.baseline_slouch = "CALIBRATING"
        self.btn_calibrate.config(text="Calibrated ✅", bg="lightgreen")

    def show_incident_report(self, raw_frame, ann_frame, ear_val, pitch_val, roll_val, current_dist, baseline):
        report_win = tk.Toplevel(self.root)
        report_win.title("Sleep Incident Report")
        report_win.geometry("900x600")
        
        raw_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        ann_rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
        
        raw_img = Image.fromarray(raw_rgb).resize((400, 300))
        ann_img = Image.fromarray(ann_rgb).resize((400, 300))
        
        raw_tk = ImageTk.PhotoImage(image=raw_img)
        ann_tk = ImageTk.PhotoImage(image=ann_img)
        
        report_win.raw_tk = raw_tk
        report_win.ann_tk = ann_tk
        
        tk.Label(report_win, text="Actual Image (Raw Camera Feed)", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=10)
        tk.Label(report_win, text="Annotated Image (System View)", font=("Arial", 12, "bold")).grid(row=0, column=1, pady=10)
        
        tk.Label(report_win, image=raw_tk).grid(row=1, column=0, padx=10)
        tk.Label(report_win, image=ann_tk).grid(row=1, column=1, padx=10)
        
        slouch_pct = (current_dist / baseline) * 100 if isinstance(baseline, (int, float)) and baseline > 0 else 100
        
        reason_text = f"--- INCIDENT LOG ---\n"
        reason_text += f"1. Eye Aspect Ratio: Dropped to {ear_val:.2f} for over {self.ear_time.get()}s.\n"
        reason_text += f"2. Head Pose: Reached Pitch: {int(pitch_val)}° / Roll: {int(roll_val)}° for over {self.hpm_time.get()}s.\n"
        reason_text += f"3. Body Posture: Vertical distance compressed to {slouch_pct:.1f}% for over {self.slouch_time.get()}s.\n"
        reason_text += f"CONCLUSION: Multi-Signal verification confirmed sleep state."
        
        tk.Label(report_win, text=reason_text, font=("Consolas", 11, "bold"), justify="left", fg="darkred", bg="#ffe6e6", padx=10, pady=10).grid(row=2, column=0, columnspan=2, pady=20, sticky="ew")
        tk.Button(report_win, text="Dismiss & Resume Monitoring", command=report_win.destroy, font=("Arial", 12), bg="lightgray", height=2).grid(row=3, column=0, columnspan=2, pady=10)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        raw_frame_copy = frame.copy()
        
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ear_triggered = hpm_triggered = slouch_triggered = False
        current_ear = current_pitch = current_roll = c_dist = 0
        ear_elapsed = hpm_elapsed = slouch_elapsed = 0.0

        face_results = face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0]
            
            if self.toggle_ear.get():
                left_ear = calculate_ear(LEFT_EYE, landmarks, w, h)
                right_ear = calculate_ear(RIGHT_EYE, landmarks, w, h)
                current_ear = (left_ear + right_ear) / 2.0
                
                for p in LEFT_EYE + RIGHT_EYE:
                    pt = (int(landmarks.landmark[p].x * w), int(landmarks.landmark[p].y * h))
                    cv2.circle(frame, pt, 2, (0, 255, 0), -1)
                
                cv2.putText(frame, f"EAR: {current_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if current_ear <= self.ear_thresh.get():
                    if self.ear_start_time is None: self.ear_start_time = time.time()
                    ear_elapsed = time.time() - self.ear_start_time
                    if ear_elapsed >= self.ear_time.get(): ear_triggered = True
                else:
                    self.ear_start_time = None

            if self.toggle_hpm.get():
                face_2d = []
                face_3d = []
                standard_3d = {1:(0,0,0), 152:(0,-330,-65), 33:(-225,170,-135), 263:(225,170,-135), 61:(-150,-150,-125), 291:(150,-150,-125)}

                for idx in FACE_3D_POINTS:
                    pt = landmarks.landmark[idx]
                    face_2d.append([int(pt.x * w), int(pt.y * h)])
                    face_3d.append(standard_3d[idx])
                
                face_2d, face_3d = np.array(face_2d, dtype=np.float64), np.array(face_3d, dtype=np.float64)
                cam_matrix = np.array([[w, 0, h/2], [0, w, w/2], [0, 0, 1]], dtype=np.float64)
                
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4, 1)))
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                
                current_pitch, current_roll = angles[0] * 360, angles[2] * 360
                cv2.putText(frame, f"Pitch: {int(current_pitch)} Roll: {int(current_roll)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                nose_pt = (int(landmarks.landmark[1].x * w), int(landmarks.landmark[1].y * h))
                p1, _ = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rot_vec, trans_vec, cam_matrix, None)
                p2, _ = cv2.projectPoints(np.array([(1000.0, 0.0, 0.0)]), rot_vec, trans_vec, cam_matrix, None)
                p3, _ = cv2.projectPoints(np.array([(0.0, 1000.0, 0.0)]), rot_vec, trans_vec, cam_matrix, None)
                
                cv2.line(frame, nose_pt, (int(p1[0][0][0]), int(p1[0][0][1])), (255, 0, 0), 3) 
                cv2.line(frame, nose_pt, (int(p2[0][0][0]), int(p2[0][0][1])), (0, 255, 0), 3) 
                cv2.line(frame, nose_pt, (int(p3[0][0][0]), int(p3[0][0][1])), (0, 0, 255), 3) 

                if abs(current_pitch) >= self.hpm_thresh.get() or abs(current_roll) >= self.hpm_thresh.get():
                    if self.head_start_time is None: self.head_start_time = time.time()
                    hpm_elapsed = time.time() - self.head_start_time
                    if hpm_elapsed >= self.hpm_time.get(): hpm_triggered = True
                else:
                    self.head_start_time = None

        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks and self.toggle_slouch.get():
            landmarks = pose_results.pose_landmarks.landmark
            nose_px = (int(landmarks[0].x * w), int(landmarks[0].y * h))
            mid_sh_px = (int((landmarks[11].x + landmarks[12].x)/2 * w), int((landmarks[11].y + landmarks[12].y)/2 * h))
            
            cv2.line(frame, nose_px, mid_sh_px, (255, 0, 255), 2)
            cv2.circle(frame, mid_sh_px, 5, (255, 0, 255), -1)
            
            c_dist = mid_sh_px[1] - nose_px[1] 
            if self.baseline_slouch == "CALIBRATING": self.baseline_slouch = c_dist
                
            if isinstance(self.baseline_slouch, (int, float)):
                if c_dist < (self.baseline_slouch * 0.8):
                    cv2.putText(frame, "SLOUCH DETECTED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if self.slouch_start_time is None: self.slouch_start_time = time.time()
                    slouch_elapsed = time.time() - self.slouch_start_time
                    if slouch_elapsed >= self.slouch_time.get(): slouch_triggered = True
                else:
                    self.slouch_start_time = None

        # --- UI Timer Updates ---
        if not self.toggle_ear.get():
            self.step_ear.config(text="✅ EAR Disabled (Bypassed)", fg="green")
            ear_triggered = True
        elif ear_triggered:
            self.step_ear.config(text=f"✅ EAR Met ({self.ear_time.get():.1f}s)", fg="green")
        elif self.ear_start_time:
            self.step_ear.config(text=f"[ ] EAR Triggering... ({ear_elapsed:.1f}s / {self.ear_time.get()}s)", fg="orange")
        else:
            self.step_ear.config(text="[ ] EAR Monitoring", fg="gray")

        if not self.toggle_hpm.get():
            self.step_hpm.config(text="✅ HPM Disabled (Bypassed)", fg="green")
            hpm_triggered = True
        elif hpm_triggered:
            self.step_hpm.config(text=f"✅ Head Pose Triggered ({self.hpm_time.get():.1f}s)", fg="green")
        elif self.head_start_time:
            self.step_hpm.config(text=f"[ ] HPM Triggering... ({hpm_elapsed:.1f}s / {self.hpm_time.get()}s)", fg="orange")
        else:
            self.step_hpm.config(text="[ ] HPM Monitoring", fg="gray")

        if not self.toggle_slouch.get():
            self.step_slouch.config(text="✅ Slouch Disabled (Bypassed)", fg="green")
            slouch_triggered = True
        elif slouch_triggered:
            self.step_slouch.config(text=f"✅ Slouching Detected ({self.slouch_time.get():.1f}s)", fg="green")
        elif self.slouch_start_time:
            self.step_slouch.config(text=f"[ ] Slouch Triggering... ({slouch_elapsed:.1f}s / {self.slouch_time.get()}s)", fg="orange")
        else:
            self.step_slouch.config(text="[ ] Posture Monitoring", fg="gray")

        if ear_triggered and hpm_triggered and slouch_triggered:
            self.alert_label.config(text="⚠️ SLEEPY DETECTED", fg="white", bg="red")
            
            if time.time() - self.last_capture_time > 10:
                self.show_incident_report(raw_frame_copy, frame.copy(), current_ear, current_pitch, current_roll, c_dist, self.baseline_slouch)
                self.last_capture_time = time.time()
                self.ear_start_time = None
                self.head_start_time = None
                self.slouch_start_time = None
        else:
            self.alert_label.config(text="STATUS: AWAKE", fg="green", bg="black")

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SleepDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()