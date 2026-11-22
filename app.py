import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import os
import shutil
import config
from face_detector import FaceDetector
from cnn_model import FaceCNNModel
from attendance_manager import AttendanceManager

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Live Attendance System")
        self.root.geometry("1300x880")
        self.root.minsize(1100, 750)
        self.root.configure(bg="#13141f")

        # Core logic models
        self.detector = FaceDetector()
        self.cnn_model = FaceCNNModel()
        self.attendance_mgr = AttendanceManager()

        # Camera state
        self.cap = None
        self.is_camera_running = False
        self.is_registering = False
        self.camera_thread = None
        self.current_frame = None

        # Registration state
        self.reg_samples_collected = 0
        self.reg_target_samples = config.SAMPLES_PER_STUDENT
        self.reg_captured_features = []
        self.last_reg_capture_time = 0

        # Build UI
        self.setup_styles()
        self.create_header()
        self.create_main_layout()

        # Connect attendance callback to update table
        self.attendance_mgr.register_listener(self.on_attendance_marked)

        # Populate registered students table immediately
        self.refresh_registered_table()

        # Start live clock
        self.update_clock()

        # Clean exit protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Configure Notebook tabs
        self.style.configure(
            "Custom.TNotebook",
            background="#1a1b26",
            borderwidth=0
        )
        self.style.configure(
            "Custom.TNotebook.Tab",
            background="#24283b",
            foreground="#a9b1d6",
            font=("Segoe UI", 10, "bold"),
            padding=[16, 6]
        )
        self.style.map(
            "Custom.TNotebook.Tab",
            background=[("selected", "#00e5ff"), ("active", "#3d59a1")],
            foreground=[("selected", "#13141f"), ("active", "#ffffff")]
        )

        # Configure Treeview table
        self.style.configure(
            "Custom.Treeview",
            background="#1e2030",
            foreground="#f0f0f5",
            rowheight=28,
            fieldbackground="#1e2030",
            font=("Segoe UI", 10),
            borderwidth=0
        )
        self.style.configure(
            "Custom.Treeview.Heading",
            background="#2a2c42",
            foreground="#00e5ff",
            font=("Segoe UI", 10, "bold"),
            relief="flat"
        )
        self.style.map("Custom.Treeview", background=[("selected", "#3d59a1")])

    def create_header(self):
        header_frame = tk.Frame(self.root, bg="#1a1b26", height=65)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        # Title
        title_box = tk.Frame(header_frame, bg="#1a1b26")
        title_box.pack(side=tk.LEFT, padx=25, pady=8)

        title_label = tk.Label(
            title_box,
            text="⚡ LIVE CNN ATTENDANCE SYSTEM",
            font=("Segoe UI", 16, "bold"),
            fg="#00e5ff",
            bg="#1a1b26"
        )
        title_label.pack(anchor="w")

        subtitle_label = tk.Label(
            title_box,
            text="Deep CNN Face Recognition (YuNet Aligned Crops + SFace 128-D Feature Embeddings)",
            font=("Segoe UI", 9),
            fg="#a9b1d6",
            bg="#1a1b26"
        )
        subtitle_label.pack(anchor="w")

        # Clock & Date (Right)
        self.clock_label = tk.Label(
            header_frame,
            text="",
            font=("Segoe UI", 11, "bold"),
            fg="#73daca",
            bg="#1a1b26"
        )
        self.clock_label.pack(side=tk.RIGHT, padx=25)

    def create_main_layout(self):
        main_container = tk.Frame(self.root, bg="#13141f")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # Left Column: Controls (Width: 380)
        self.left_panel = tk.Frame(main_container, bg="#13141f", width=380)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        self.left_panel.pack_propagate(False)

        # Right Column: Camera + Live Attendance / Students Table
        self.right_panel = tk.Frame(main_container, bg="#13141f")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.build_registration_card()
        self.build_model_card()
        self.build_controls_card()
        self.build_camera_view()
        self.build_tables_notebook()

    def build_registration_card(self):
        card = tk.LabelFrame(
            self.left_panel,
            text=" 👤 Student Registration ",
            font=("Segoe UI", 11, "bold"),
            fg="#7aa2f7",
            bg="#1a1b26",
            bd=1,
            relief="solid",
            padx=15,
            pady=12
        )
        card.pack(fill=tk.X, pady=(0, 15))

        # Student Name
        tk.Label(card, text="Student Full Name:", font=("Segoe UI", 9, "bold"), fg="#c0caf5", bg="#1a1b26").pack(anchor="w")
        self.name_entry = tk.Entry(card, font=("Segoe UI", 10), bg="#24283b", fg="#ffffff", insertbackground="#ffffff", relief="flat")
        self.name_entry.pack(fill=tk.X, pady=(4, 10), ipady=5)

        # Roll Number
        tk.Label(card, text="Roll Number / Student ID:", font=("Segoe UI", 9, "bold"), fg="#c0caf5", bg="#1a1b26").pack(anchor="w")
        self.roll_entry = tk.Entry(card, font=("Segoe UI", 10), bg="#24283b", fg="#ffffff", insertbackground="#ffffff", relief="flat")
        self.roll_entry.pack(fill=tk.X, pady=(4, 12), ipady=5)

        # Register Button
        self.reg_btn = tk.Button(
            card,
            text="📸 Capture & Register Face",
            font=("Segoe UI", 10, "bold"),
            bg="#2ac3de",
            fg="#13141f",
            activebackground="#7dcfff",
            activeforeground="#13141f",
            relief="flat",
            cursor="hand2",
            command=self.start_registration
        )
        self.reg_btn.pack(fill=tk.X, ipady=6)

        # Progress bar
        self.reg_progress = ttk.Progressbar(card, orient="horizontal", mode="determinate", maximum=self.reg_target_samples)
        self.reg_progress.pack(fill=tk.X, pady=(10, 4))

        self.reg_status_label = tk.Label(
            card,
            text="Ready to register new student",
            font=("Segoe UI", 8, "italic"),
            fg="#a9b1d6",
            bg="#1a1b26"
        )
        self.reg_status_label.pack(anchor="w")

    def build_model_card(self):
        card = tk.LabelFrame(
            self.left_panel,
            text=" 🧠 CNN Model Status ",
            font=("Segoe UI", 11, "bold"),
            fg="#bb9af7",
            bg="#1a1b26",
            bd=1,
            relief="solid",
            padx=15,
            pady=12
        )
        card.pack(fill=tk.X, pady=(0, 15))

        self.enrolled_badge = tk.Label(
            card,
            text="Registered Students: 0",
            font=("Segoe UI", 10, "bold"),
            fg="#73daca",
            bg="#1a1b26"
        )
        self.enrolled_badge.pack(anchor="w", pady=(0, 6))

        info_label = tk.Label(
            card,
            text="Detector: YuNet (5 Facial Landmarks Alignment)\nRecognizer: Deep SFace CNN (128-D Embeddings)",
            font=("Segoe UI", 8),
            fg="#7aa2f7",
            bg="#1a1b26",
            justify="left"
        )
        info_label.pack(anchor="w", pady=(0, 10))

        train_btn = tk.Button(
            card,
            text="⚡ Re-train / Sync CNN Embeddings",
            font=("Segoe UI", 9, "bold"),
            bg="#3b4261",
            fg="#c0caf5",
            activebackground="#414868",
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            command=self.retrain_model_threaded
        )
        train_btn.pack(fill=tk.X, ipady=4)

    def build_controls_card(self):
        card = tk.LabelFrame(
            self.left_panel,
            text=" ⚙️ Attendance Controls ",
            font=("Segoe UI", 11, "bold"),
            fg="#ff9e64",
            bg="#1a1b26",
            bd=1,
            relief="solid",
            padx=15,
            pady=12
        )
        card.pack(fill=tk.BOTH, expand=True)

        # Primary Live Attendance Toggle
        self.attendance_toggle_btn = tk.Button(
            card,
            text="▶  START LIVE ATTENDANCE",
            font=("Segoe UI", 11, "bold"),
            bg="#73daca",
            fg="#13141f",
            activebackground="#9ece6a",
            activeforeground="#13141f",
            relief="flat",
            cursor="hand2",
            command=self.toggle_live_attendance
        )
        self.attendance_toggle_btn.pack(fill=tk.X, ipady=8, pady=(0, 12))

        # Confidence Threshold Slider
        tk.Label(card, text="Match Sensitivity (Cosine Threshold):", font=("Segoe UI", 9, "bold"), fg="#c0caf5", bg="#1a1b26").pack(anchor="w")
        initial_pct = int(config.SIMILARITY_THRESHOLD * 100)
        self.threshold_val_label = tk.Label(card, text=f"{initial_pct}%", font=("Segoe UI", 9), fg="#00e5ff", bg="#1a1b26")
        self.threshold_val_label.pack(anchor="e")

        self.threshold_slider = ttk.Scale(card, from_=25, to=65, value=initial_pct, command=self.on_threshold_change)
        self.threshold_slider.pack(fill=tk.X, pady=(0, 15))

        # Export CSV Button
        export_btn = tk.Button(
            card,
            text="📥 Export Attendance to CSV",
            font=("Segoe UI", 9, "bold"),
            bg="#24283b",
            fg="#00e5ff",
            relief="flat",
            cursor="hand2",
            command=self.export_csv
        )
        export_btn.pack(fill=tk.X, ipady=5, pady=(0, 8))

        # Clear Session Button
        clear_btn = tk.Button(
            card,
            text="🗑 Clear Session Attendance",
            font=("Segoe UI", 9, "bold"),
            bg="#24283b",
            fg="#f7768e",
            relief="flat",
            cursor="hand2",
            command=self.clear_session
        )
        clear_btn.pack(fill=tk.X, ipady=5)

    def build_camera_view(self):
        cam_frame = tk.Frame(self.right_panel, bg="#1a1b26", bd=1, relief="solid", height=380)
        cam_frame.pack(fill=tk.X, pady=(0, 15))
        cam_frame.pack_propagate(False)

        self.cam_label = tk.Label(
            cam_frame,
            text="📷 CAMERA OFFLINE\nClick 'START LIVE ATTENDANCE' or 'Capture & Register Face' to activate.",
            font=("Segoe UI", 12),
            fg="#7aa2f7",
            bg="#16161e"
        )
        self.cam_label.pack(fill=tk.BOTH, expand=True)

    def build_tables_notebook(self):
        self.notebook = ttk.Notebook(self.right_panel, style="Custom.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Live Attendance Log
        tab_attendance = tk.Frame(self.notebook, bg="#1a1b26")
        self.notebook.add(tab_attendance, text=" 📋 Live Attendance Log ")

        tbl_header = tk.Frame(tab_attendance, bg="#1a1b26")
        tbl_header.pack(fill=tk.X, padx=15, pady=6)

        tk.Label(
            tbl_header,
            text="Live Session Records (Marked automatically when student faces camera):",
            font=("Segoe UI", 9, "italic"),
            fg="#a9b1d6",
            bg="#1a1b26"
        ).pack(side=tk.LEFT)

        self.present_counter_badge = tk.Label(
            tbl_header,
            text="Total Present: 0",
            font=("Segoe UI", 10, "bold"),
            fg="#73daca",
            bg="#24283b",
            padx=10,
            pady=3
        )
        self.present_counter_badge.pack(side=tk.RIGHT)

        columns = ("sr", "roll", "name", "date", "time", "confidence", "status")
        self.tree = ttk.Treeview(
            tab_attendance,
            columns=columns,
            show="headings",
            style="Custom.Treeview",
            selectmode="browse"
        )

        self.tree.heading("sr", text="Sr #")
        self.tree.heading("roll", text="Roll No")
        self.tree.heading("name", text="Student Name")
        self.tree.heading("date", text="Date")
        self.tree.heading("time", text="Entry Time")
        self.tree.heading("confidence", text="Match %")
        self.tree.heading("status", text="Status")

        self.tree.column("sr", width=45, anchor="center")
        self.tree.column("roll", width=100, anchor="center")
        self.tree.column("name", width=180, anchor="w")
        self.tree.column("date", width=95, anchor="center")
        self.tree.column("time", width=110, anchor="center")
        self.tree.column("confidence", width=85, anchor="center")
        self.tree.column("status", width=90, anchor="center")

        sc1 = ttk.Scrollbar(tab_attendance, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=sc1.set)
        sc1.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 10))

        # Tab 2: Registered Students Database
        tab_registered = tk.Frame(self.notebook, bg="#1a1b26")
        self.notebook.add(tab_registered, text=" 👥 Registered Students Database ")

        reg_header = tk.Frame(tab_registered, bg="#1a1b26")
        reg_header.pack(fill=tk.X, padx=15, pady=6)

        tk.Label(
            reg_header,
            text="All students saved and enrolled in the CNN model:",
            font=("Segoe UI", 9, "italic"),
            fg="#a9b1d6",
            bg="#1a1b26"
        ).pack(side=tk.LEFT)

        del_btn = tk.Button(
            reg_header,
            text="🗑 Delete Selected Student",
            font=("Segoe UI", 8, "bold"),
            bg="#24283b",
            fg="#f7768e",
            relief="flat",
            cursor="hand2",
            command=self.delete_selected_student
        )
        del_btn.pack(side=tk.RIGHT)

        clear_all_btn = tk.Button(
            reg_header,
            text="⚠️ Delete All Data",
            font=("Segoe UI", 8, "bold"),
            bg="#24283b",
            fg="#ff5555",
            relief="flat",
            cursor="hand2",
            command=self.delete_all_data
        )
        clear_all_btn.pack(side=tk.RIGHT, padx=8)

        reg_cols = ("roll", "name", "samples", "status")
        self.reg_tree = ttk.Treeview(
            tab_registered,
            columns=reg_cols,
            show="headings",
            style="Custom.Treeview",
            selectmode="browse"
        )

        self.reg_tree.heading("roll", text="Roll Number / ID")
        self.reg_tree.heading("name", text="Student Full Name")
        self.reg_tree.heading("samples", text="Aligned Face Samples")
        self.reg_tree.heading("status", text="Model Status")

        self.reg_tree.column("roll", width=120, anchor="center")
        self.reg_tree.column("name", width=220, anchor="w")
        self.reg_tree.column("samples", width=160, anchor="center")
        self.reg_tree.column("status", width=120, anchor="center")

        sc2 = ttk.Scrollbar(tab_registered, orient=tk.VERTICAL, command=self.reg_tree.yview)
        self.reg_tree.configure(yscroll=sc2.set)
        sc2.pack(side=tk.RIGHT, fill=tk.Y)
        self.reg_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 10))

    def refresh_registered_table(self):
        """Refreshes the Registered Students table from the CNN database."""
        for item in self.reg_tree.get_children():
            self.reg_tree.delete(item)

        students = self.cnn_model.get_enrolled_list()
        for s in students:
            self.reg_tree.insert(
                "",
                tk.END,
                values=(s["roll_no"], s["name"], f"{s['samples']} Samples", "Ready for Live")
            )

        count = len(students)
        self.enrolled_badge.config(text=f"Registered Students: {count}")

    def delete_selected_student(self):
        selected = self.reg_tree.selection()
        if not selected:
            messagebox.showwarning("Selection Required", "Please select a student from the table to delete.")
            return

        item = self.reg_tree.item(selected[0])
        roll_no = str(item["values"][0])
        name = str(item["values"][1])

        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to remove student {name} (Roll: {roll_no}) from the CNN database?"
        )
        if confirm:
            if roll_no in self.cnn_model.student_database:
                del self.cnn_model.student_database[roll_no]
                self.cnn_model.save_embeddings()

            for folder in os.listdir(config.DATASET_DIR):
                if folder.startswith(f"{roll_no}_") or folder == roll_no:
                    fpath = os.path.join(config.DATASET_DIR, folder)
                    if os.path.isdir(fpath):
                        shutil.rmtree(fpath, ignore_errors=True)

            self.refresh_registered_table()
            messagebox.showinfo("Student Removed", f"Student {name} removed successfully.")

    def delete_all_data(self):
        confirm = messagebox.askyesno(
            "Delete All Data",
            "Are you sure you want to completely erase ALL student datasets and model embeddings?"
        )
        if confirm:
            self.cnn_model.student_database.clear()
            self.cnn_model.save_embeddings()

            if os.path.exists(config.DATASET_DIR):
                shutil.rmtree(config.DATASET_DIR, ignore_errors=True)
            os.makedirs(config.DATASET_DIR, exist_ok=True)

            self.refresh_registered_table()
            messagebox.showinfo("Cleared", "All student data has been erased. You can now register fresh data.")

    def on_threshold_change(self, val):
        pct = float(val) / 100.0
        self.cnn_model.threshold = pct
        self.threshold_val_label.config(text=f"{int(pct * 100)}%")

    def update_clock(self):
        now = datetime.now()
        self.clock_label.config(text=now.strftime("%A, %d %b %Y  |  %I:%M:%S %p"))
        self.root.after(1000, self.update_clock)

    # ================= STUDENT REGISTRATION =================

    def start_registration(self):
        name = self.name_entry.get().strip()
        roll_no = self.roll_entry.get().strip()

        if not name or not roll_no:
            messagebox.showwarning("Incomplete Data", "Please enter both Student Name and Roll Number!")
            return

        if self.is_camera_running:
            self.stop_live_attendance()

        self.is_registering = True
        self.reg_samples_collected = 0
        self.reg_captured_features = []
        self.reg_btn.config(state=tk.DISABLED)
        self.attendance_toggle_btn.config(state=tk.DISABLED)
        self.reg_progress["value"] = 0
        self.reg_status_label.config(text="Initializing camera for face capture...", fg="#7aa2f7")

        threading.Thread(target=self._run_registration_loop, args=(name, roll_no), daemon=True).start()

    def _run_registration_loop(self, name, roll_no):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("Camera Error", "Could not open webcam."))
            self.root.after(0, self._reset_registration_ui)
            return

        student_dir = os.path.join(config.DATASET_DIR, f"{roll_no}_{name.replace(' ', '_')}")
        os.makedirs(student_dir, exist_ok=True)

        last_capture_time = 0

        while self.is_registering and self.reg_samples_collected < self.reg_target_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            faces = self.detector.detect_faces(frame)

            current_time = time.time()
            if len(faces) == 1:
                face_item = faces[0]
                bbox = face_item["bbox"]
                self.detector.draw_futuristic_box(frame, bbox, color=(0, 255, 128))

                if current_time - last_capture_time >= 0.15:
                    feat, aligned = self.cnn_model.extract_from_frame(frame, face_item["raw"])
                    if feat is not None and aligned is not None:
                        self.reg_captured_features.append(feat)
                        self.reg_samples_collected += 1
                        last_capture_time = current_time

                        # Save aligned face crop
                        crop_path = os.path.join(student_dir, f"sample_{self.reg_samples_collected:03d}.jpg")
                        cv2.imwrite(crop_path, aligned)

                        # Update UI
                        pct = int((self.reg_samples_collected / self.reg_target_samples) * 100)
                        self.root.after(0, self._update_reg_progress, self.reg_samples_collected, pct)

            self.root.after(0, self._display_frame, frame)
            time.sleep(0.02)

        cap.release()

        if self.reg_samples_collected >= self.reg_target_samples:
            self.root.after(0, lambda: self.reg_status_label.config(text="Saving CNN embeddings...", fg="#ff9e64"))
            success, msg = self.cnn_model.enroll_student(roll_no, name, self.reg_captured_features)
            self.root.after(0, lambda: self._on_registration_finished(success, msg, name, roll_no))
        else:
            self.root.after(0, self._reset_registration_ui)

    def _update_reg_progress(self, count, pct):
        self.reg_progress["value"] = count
        self.reg_status_label.config(text=f"Capturing: {count}/{self.reg_target_samples} ({pct}%)", fg="#73daca")

    def _on_registration_finished(self, success, msg, name, roll_no):
        self._reset_registration_ui()
        if success:
            self.refresh_registered_table()
            self.name_entry.delete(0, tk.END)
            self.roll_entry.delete(0, tk.END)
            self.notebook.select(1)
            messagebox.showinfo(
                "Data Saved Successfully!",
                f"Student: {name}\nRoll No: {roll_no}\n\nAligned Face Samples & 128-D CNN Embeddings saved!\nThe student is now ready for Live Attendance."
            )
        else:
            messagebox.showerror("Registration Error", msg)

    def _reset_registration_ui(self):
        self.is_registering = False
        self.reg_btn.config(state=tk.NORMAL)
        self.attendance_toggle_btn.config(state=tk.NORMAL)
        self.reg_progress["value"] = 0
        self.reg_status_label.config(text="Ready to register new student", fg="#a9b1d6")
        self.cam_label.config(image="", text="📷 CAMERA IDLE\nReady for Live Attendance.")

    # ================= LIVE ATTENDANCE =================

    def toggle_live_attendance(self):
        if self.is_camera_running:
            self.stop_live_attendance()
        else:
            self.start_live_attendance()

    def start_live_attendance(self):
        if self.cnn_model.get_enrolled_count() == 0:
            resp = messagebox.askyesno(
                "No Students Enrolled",
                "There are no registered students in the database.\nDo you want to start the camera to test Unknown face detection?"
            )
            if not resp:
                return

        self.is_camera_running = True
        self.attendance_toggle_btn.config(
            text="⏹  STOP LIVE ATTENDANCE",
            bg="#f7768e",
            activebackground="#ff9e64"
        )
        self.reg_btn.config(state=tk.DISABLED)

        self.notebook.select(0)

        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera!")
            self.stop_live_attendance()
            return

        threading.Thread(target=self._live_attendance_loop, daemon=True).start()

    def stop_live_attendance(self):
        self.is_camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.attendance_toggle_btn.config(
            text="▶  START LIVE ATTENDANCE",
            bg="#73daca",
            activebackground="#9ece6a"
        )
        self.reg_btn.config(state=tk.NORMAL)
        self.cam_label.config(image="", text="📷 CAMERA STOPPED\nClick 'START LIVE ATTENDANCE' to resume.")

    def _live_attendance_loop(self):
        while self.is_camera_running and self.cap is not None:
            ret, self.cap_read_frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(self.cap_read_frame, 1)
            faces = self.detector.detect_faces(frame)

            for face_item in faces:
                bbox = face_item["bbox"]
                feat, aligned = self.cnn_model.extract_from_frame(frame, face_item["raw"])
                if feat is None:
                    continue

                is_match, student, confidence = self.cnn_model.match_face(feat)

                if is_match:
                    roll = student["roll_no"]
                    name = student["name"]
                    newly_marked, status_msg, rec = self.attendance_mgr.mark_attendance(roll, name, confidence)

                    if newly_marked:
                        self.detector.draw_hud(frame, bbox, name, roll, confidence, "PRESENT", is_marked=False)
                    else:
                        self.detector.draw_hud(frame, bbox, name, roll, confidence, status_msg, is_marked=True)
                else:
                    self.detector.draw_hud(frame, bbox, "Unknown", "N/A", confidence, "UNREGISTERED", is_marked=False)

            self.root.after(0, self._display_frame, frame)
            time.sleep(0.015)

    def _display_frame(self, frame):
        """Renders OpenCV BGR frame on Tkinter Label widget."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]

        target_w = 640
        target_h = int((target_w / w) * h)
        resized = cv2.resize(rgb_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        img = Image.fromarray(resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.cam_label.imgtk = imgtk
        self.cam_label.configure(image=imgtk, text="")

    def on_attendance_marked(self, record):
        """Called automatically whenever a new attendance entry is recorded."""
        self.root.after(0, self._insert_table_row, record)

    def _insert_table_row(self, r):
        self.tree.insert(
            "",
            0,
            values=(r["index"], r["roll_no"], r["name"], r["date"], r["time"], r["confidence"], r["status"])
        )
        self.present_counter_badge.config(text=f"Total Present: {self.attendance_mgr.get_total_present()}")

    def clear_session(self):
        if messagebox.askyesno("Clear Session", "Are you sure you want to clear current live attendance records?"):
            self.attendance_mgr.clear_session()
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.present_counter_badge.config(text="Total Present: 0")

    def export_csv(self):
        if self.attendance_mgr.get_total_present() == 0:
            messagebox.showinfo("Export CSV", "No attendance records to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile=f"Attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if file_path:
            ok, msg = self.attendance_mgr.export_to_csv(file_path)
            if ok:
                messagebox.showinfo("Export Successful", msg)
            else:
                messagebox.showerror("Export Failed", msg)

    def retrain_model_threaded(self):
        threading.Thread(target=self._retrain_job, daemon=True).start()

    def _retrain_job(self):
        from train_model import train_or_rebuild_embeddings
        success = train_or_rebuild_embeddings()
        self.cnn_model.load_embeddings()
        self.root.after(0, self.refresh_registered_table)
        if success:
            self.root.after(0, lambda: messagebox.showinfo("Model Ready", "CNN Embeddings successfully rebuilt and synchronized!"))
        else:
            self.root.after(0, lambda: messagebox.showwarning("Training Info", "No student datasets to train."))

    def on_closing(self):
        self.is_camera_running = False
        self.is_registering = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
