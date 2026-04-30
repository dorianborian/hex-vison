import cv2
import numpy as np
import mss
import time
import threading
import json
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
import customtkinter as ctk
from ultralytics import YOLO
from PIL import Image, ImageTk

class RegionSelector(tk.Toplevel):
    def __init__(self, master, callback):
        super().__init__(master)
        self.callback = callback
        self.attributes('-alpha', 0.4)
        self.attributes('-fullscreen', True)
        self.attributes('-topmost', True)
        self.configure(cursor="cross")

        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.start_x = None
        self.start_y = None
        self.rect = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Escape>", lambda e: self.destroy())

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2, fill="gray")

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        end_x = event.x
        end_y = event.y

        left = min(self.start_x, end_x)
        top = min(self.start_y, end_y)
        width = abs(end_x - self.start_x)
        height = abs(end_y - self.start_y)

        if width > 10 and height > 10:
            self.callback({"top": top, "left": left, "width": width, "height": height})
            
        self.destroy()

class HexVisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        cv2.setUseOptimized(True)
        try:
            cv2.ocl.setUseOpenCL(True)
        except Exception:
            pass
        self.title("Hex-Vision")
        self.geometry("800x800")
        self.resizable(False, False)
        self.minsize(800, 800)
        self.maxsize(800, 800)
        self._viz_stacked = None
        self.last_fwd_mag = 0.0
        self.last_turn_mag = 0.0
        self.mouse_hold_delay_sec = 1.0
        self.mouse_hold_arm_time = None
        self.mouse_left_is_down = False
        self.mode_click_cooldown_sec = 0.6
        self.last_mode_click_time = 0.0
        self.turn_comp_gain = 0.55
        self.follow_persist_sec = 5.0
        self.follow_turn_deadzone_px = 40.0
        self.follow_turn_gain = 1.30
        self.follow_turn_min_cmd = 0.06
        self.follow_turn_rate_limit = 0.18
        self.follow_persist_decay = 0.86
        self.follow_persist_turn_boost = 0.80
        self.autonomy_enabled = ctk.BooleanVar(value=False)
        self.last_live_frame = None
        self.persist_turn_only = ctk.BooleanVar(value=False)

        # Inference tuning for faster FPS
        self.infer_imgsz = 320
        self.infer_conf = 0.40
        self.infer_iou = 0.50
        self.infer_max_det = 6

        # state
        self.is_running = False
        self.capture_thread = None
        
        # Robot Goal states
        self.active_goal = "Follow Object"
        self.target_object = "person"
        self.max_fwd_pct = 1.0
        self.max_rev_pct = 1.0
        self.max_turn_left_pct = 0.25
        self.max_turn_right_pct = 0.25
        self.looking_mode = False
        self.looking_mode_requested = False
        self.follow_look_state = "idle"
        self.follow_cleanup_state = "idle"
        
        # default screen regions
        self.rgb_monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
        self.depth_monitor = None # User must set this if they want depth scanning
        self.controller_monitor = None # User must set this to output motion vectors
        self.translate_motion = False
        self.uv_spot = None
        self.tripod_spot = None
        self.hold_position_spot = None
        self.rear_most_spot = None
        self.zw_spot = None
        self.focus_window_spot = None
        
        # Load YOLOv8 segmentation model
        self.model = None

        self.setup_ui()
        self.load_saved_spots()

    def setup_ui(self):
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=3, pady=3)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(1, weight=1)
        self.main_container.grid_rowconfigure(1, weight=1)

        self.label = ctk.CTkLabel(self.main_container, text="Hex-Vision Object Tracker", font=ctk.CTkFont(size=15, weight="bold"))
        self.label.grid(row=0, column=0, columnspan=2, padx=6, pady=(2, 3), sticky="ew")

        # Bottom dashboard area split into left/right columns
        self.bottom_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.bottom_frame.grid(row=1, column=0, columnspan=2, padx=4, pady=(0, 1), sticky="nsew")
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(1, weight=1)
        self.bottom_frame.grid_rowconfigure(0, weight=1)

        self.left_panel = ctk.CTkFrame(self.bottom_frame, fg_color="transparent")
        self.left_panel.grid(row=0, column=0, padx=(0, 4), sticky="nsew")

        self.right_panel = ctk.CTkScrollableFrame(self.bottom_frame, fg_color="transparent")
        self.right_panel.grid(row=0, column=1, padx=(4, 0), sticky="nsew")

        # Embedded live output (replaces external OpenCV window) on right side only.
        self.output_frame = ctk.CTkFrame(self.right_panel, height=230)
        self.output_frame.pack(fill="x", pady=(0, 3))
        self.output_frame.pack_propagate(False)
        self.output_frame.grid_propagate(False)
        self.output_frame.grid_rowconfigure(1, weight=1)
        self.output_frame.grid_columnconfigure(0, weight=1)

        self.output_title = ctk.CTkLabel(self.output_frame, text="Live Vision Output", font=ctk.CTkFont(size=12, weight="bold"))
        self.output_title.grid(row=0, column=0, padx=6, pady=(3, 1), sticky="w")

        self.output_view = tk.Label(self.output_frame, bg="#101010")
        self.output_view.grid(row=1, column=0, padx=4, pady=(0, 4), sticky="nsew")
        self.output_imgtk = None

        # Region Frame
        self.region_frame = ctk.CTkFrame(self.left_panel)
        self.region_frame.pack(fill="x", pady=(0, 2))

        self.rgb_region_label = ctk.CTkLabel(self.region_frame, text=f"RGB Zone: {self.rgb_monitor['width']}x{self.rgb_monitor['height']} at ({self.rgb_monitor['left']}, {self.rgb_monitor['top']})", font=ctk.CTkFont(size=10))
        self.rgb_region_label.pack(pady=(2, 0))

        self.btn_set_rgb = ctk.CTkButton(self.region_frame, text="Set RGB Camera Zone", command=self.set_rgb_region, height=24)
        self.btn_set_rgb.pack(pady=1)

        self.depth_region_label = ctk.CTkLabel(self.region_frame, text="Depth Zone: Not Set", font=ctk.CTkFont(size=10))
        self.depth_region_label.pack(pady=(1, 0))

        self.btn_set_depth = ctk.CTkButton(self.region_frame, text="Set Depth Camera Zone", command=self.set_depth_region, height=24)
        self.btn_set_depth.pack(pady=1)
        
        self.btn_set_controller = ctk.CTkButton(self.region_frame, text="Set Controller Output Region", command=self.set_controller_region, height=24)
        self.btn_set_controller.pack(pady=1)

        self.spot_table = ctk.CTkFrame(self.region_frame)
        self.spot_table.pack(fill="x", padx=5, pady=(3, 2))
        self.spot_table.columnconfigure(0, weight=1)
        self.spot_table.columnconfigure(1, weight=0)
        self.spot_table.columnconfigure(2, weight=1)
        self.spot_table.columnconfigure(3, weight=0)

        ctk.CTkLabel(self.spot_table, text="Spot Setup", font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=0, columnspan=4, pady=(2, 2), sticky="w")

        self.lbl_uv_spot = ctk.CTkLabel(self.spot_table, text="UV Spot: Not Set", anchor="w", font=ctk.CTkFont(size=10))
        self.lbl_uv_spot.grid(row=1, column=0, padx=(4, 4), pady=1, sticky="ew")
        self.btn_set_uv_spot = ctk.CTkButton(self.spot_table, text="Set", width=48, height=22, command=self.set_uv_spot)
        self.btn_set_uv_spot.grid(row=1, column=1, padx=(0, 6), pady=1)

        self.lbl_tripod_spot = ctk.CTkLabel(self.spot_table, text="Tripod Spot: Not Set", anchor="w", font=ctk.CTkFont(size=10))
        self.lbl_tripod_spot.grid(row=1, column=2, padx=(4, 4), pady=1, sticky="ew")
        self.btn_set_tripod_spot = ctk.CTkButton(self.spot_table, text="Set", width=48, height=22, command=self.set_tripod_spot)
        self.btn_set_tripod_spot.grid(row=1, column=3, padx=(0, 4), pady=1)

        self.lbl_hold_pos_spot = ctk.CTkLabel(self.spot_table, text="Hold Position: Not Set", anchor="w", font=ctk.CTkFont(size=10))
        self.lbl_hold_pos_spot.grid(row=2, column=0, padx=(4, 4), pady=1, sticky="ew")
        self.btn_set_hold_pos_spot = ctk.CTkButton(self.spot_table, text="Set", width=48, height=22, command=self.set_hold_position_spot)
        self.btn_set_hold_pos_spot.grid(row=2, column=1, padx=(0, 6), pady=1)

        self.lbl_rear_spot = ctk.CTkLabel(self.spot_table, text="Rear-most: Not Set", anchor="w", font=ctk.CTkFont(size=10))
        self.lbl_rear_spot.grid(row=2, column=2, padx=(4, 4), pady=1, sticky="ew")
        self.btn_set_rear_spot = ctk.CTkButton(self.spot_table, text="Set", width=48, height=22, command=self.set_rear_most_spot)
        self.btn_set_rear_spot.grid(row=2, column=3, padx=(0, 4), pady=1)

        self.lbl_zw_spot = ctk.CTkLabel(self.spot_table, text="ZW Spot: Not Set", anchor="w", font=ctk.CTkFont(size=10))
        self.lbl_zw_spot.grid(row=3, column=0, padx=(4, 4), pady=(1, 2), sticky="ew")
        self.btn_set_zw_spot = ctk.CTkButton(self.spot_table, text="Set", width=48, height=22, command=self.set_zw_spot)
        self.btn_set_zw_spot.grid(row=3, column=1, padx=(0, 6), pady=(1, 2))

        self.lbl_focus_spot = ctk.CTkLabel(self.spot_table, text="Focus Window: Not Set", anchor="w", font=ctk.CTkFont(size=10))
        self.lbl_focus_spot.grid(row=3, column=2, padx=(4, 4), pady=(1, 2), sticky="ew")
        self.btn_set_focus_spot = ctk.CTkButton(self.spot_table, text="Set", width=48, height=22, command=self.set_focus_window_spot)
        self.btn_set_focus_spot.grid(row=3, column=3, padx=(0, 4), pady=(1, 2))

        self.btn_save_spots = ctk.CTkButton(self.spot_table, text="Save Spots", height=24, command=self.save_spots)
        self.btn_save_spots.grid(row=4, column=0, columnspan=4, padx=4, pady=(1, 3), sticky="ew")

        # Control Frame
        self.control_frame = ctk.CTkFrame(self.left_panel)
        self.control_frame.pack(fill="x", pady=(0, 2))

        self.autonomy_frame = ctk.CTkFrame(self.control_frame, fg_color="#1c2b3a")
        self.autonomy_frame.pack(fill="x", padx=3, pady=(3, 1))
        self.autonomy_frame.columnconfigure(0, weight=1)

        self.autonomy_title = ctk.CTkLabel(
            self.autonomy_frame,
            text="Autonomy",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#c9d6e6",
        )
        self.autonomy_title.grid(row=0, column=0, padx=6, pady=(4, 1), sticky="w")

        self.autonomy_btn = ctk.CTkButton(
            self.autonomy_frame,
            text="Autonomous Input: OFF",
            command=self.toggle_controller,
            font=ctk.CTkFont(size=11, weight="bold"),
            height=32,
            corner_radius=4,
        )
        self.autonomy_btn.grid(row=1, column=0, padx=6, pady=(0, 4), sticky="ew")
        self.update_autonomy_visual()

        self.lbl_looking_debug = ctk.CTkLabel(
            self.autonomy_frame,
            text="dbg req:OFF act:OFF seq:idle",
            font=ctk.CTkFont(size=9),
            text_color="gray70",
            anchor="e",
        )
        self.lbl_looking_debug.grid(row=2, column=0, padx=6, pady=(0, 4), sticky="e")

        self.btn_start = ctk.CTkButton(self.control_frame, text="Start Vision", fg_color="green", hover_color="darkgreen", height=24, command=self.start_vision)
        self.btn_start.pack(side="left", padx=4, pady=3, expand=True)

        self.btn_stop = ctk.CTkButton(self.control_frame, text="Stop Vision", fg_color="red", hover_color="darkred", state="disabled", height=24, command=self.stop_vision)
        self.btn_stop.pack(side="right", padx=4, pady=3, expand=True)

        # Robot Goals Frame
        self.goals_frame = ctk.CTkFrame(self.left_panel)
        self.goals_frame.pack(fill="x")
        
        ctk.CTkLabel(self.goals_frame, text="Robot Goals & Directives", font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=0, columnspan=2, pady=1)
        
        self.goal_var = ctk.StringVar(value="Follow Object")
        self.opt_goal = ctk.CTkOptionMenu(self.goals_frame, variable=self.goal_var, 
                                          values=["Avoid All Objects", "Follow Object", "Search for Object"], 
                                          command=self.set_active_goal, height=24)
        self.opt_goal.grid(row=1, column=0, padx=5, pady=1, sticky="ew")
        
        self.entry_target = ctk.CTkEntry(self.goals_frame, placeholder_text="Target Object/Class (e.g. person)", height=24)
        self.entry_target.insert(0, "person")
        self.entry_target.configure(state="disabled")
        self.entry_target.grid(row=1, column=1, padx=5, pady=1, sticky="ew")
        
        self.motion_limits_frame = ctk.CTkFrame(self.goals_frame, fg_color="transparent")
        self.motion_limits_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=(1, 1), sticky="ew")
        self.motion_limits_frame.columnconfigure(0, weight=1)
        self.motion_limits_frame.columnconfigure(1, weight=1)

        # Motion limits (speed + turn)
        self.fwd_sld_label = ctk.CTkLabel(self.motion_limits_frame, text="Max Speed: 100%", font=ctk.CTkFont(size=9))
        self.fwd_sld_label.grid(row=0, column=0, padx=(0, 6), pady=(0, 0), sticky="w")
        self.turn_sld_label = ctk.CTkLabel(self.motion_limits_frame, text=f"Max Turn: {int(self.max_turn_left_pct * 100)}%", font=ctk.CTkFont(size=9))
        self.turn_sld_label.grid(row=0, column=1, padx=(6, 0), pady=(0, 0), sticky="w")

        self.fwd_sld = ctk.CTkSlider(self.motion_limits_frame, from_=0, to=100, command=self.update_fwd_limit)
        self.fwd_sld.set(100)
        self.fwd_sld.grid(row=1, column=0, padx=(0, 6), pady=(1, 0), sticky="ew")

        self.turn_sld = ctk.CTkSlider(self.motion_limits_frame, from_=0, to=100, command=self.update_turn_limit)
        self.turn_sld.set(self.max_turn_left_pct * 100)
        self.turn_sld.grid(row=1, column=1, padx=(6, 0), pady=(1, 0), sticky="ew")

        self.deadzone_sld_label = ctk.CTkLabel(self.goals_frame, text=f"Turn Deadzone: {int(self.follow_turn_deadzone_px)} px", font=ctk.CTkFont(size=9))
        self.deadzone_sld_label.grid(row=3, column=0, columnspan=2, padx=5, pady=(1, 0), sticky="w")
        self.deadzone_sld = ctk.CTkSlider(self.goals_frame, from_=0, to=140, command=self.update_deadzone_limit)
        self.deadzone_sld.set(self.follow_turn_deadzone_px)
        self.deadzone_sld.grid(row=4, column=0, columnspan=2, padx=5, pady=(0, 2), sticky="ew")

        self.persist_turn_only_chk = ctk.CTkCheckBox(
            self.goals_frame,
            text="Persistence: Turn Only",
            variable=self.persist_turn_only,
            command=self.on_persist_mode_toggle,
            font=ctk.CTkFont(size=10),
        )
        self.persist_turn_only_chk.grid(row=5, column=0, columnspan=2, padx=5, pady=(0, 3), sticky="w")
        
        self.goals_frame.columnconfigure(0, weight=1)
        self.goals_frame.columnconfigure(1, weight=1)
        self.update_looking_debug_line()

        # Telemetry Data Frame
        self.telemetry_frame = ctk.CTkFrame(self.right_panel)
        self.telemetry_frame.pack(fill="x", expand=False, pady=(0, 3))

        ctk.CTkLabel(self.telemetry_frame, text="Robot Brain Telemetry", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=3)

        self.lbl_directive = ctk.CTkLabel(self.telemetry_frame, text="CURRENT DIRECTIVE:\n[ STANDBY ]", font=ctk.CTkFont(size=11, weight="bold"), text_color="cyan")
        self.lbl_directive.pack(pady=4)

        self.lbl_threat_matrix = ctk.CTkLabel(self.telemetry_frame, text="THREAT MATRIX\nL: 0   |   C: 0   |   R: 0", font=ctk.CTkFont(size=11))
        self.lbl_threat_matrix.pack(pady=2)
        
        self.lbl_perf = ctk.CTkLabel(self.telemetry_frame, text="FPS: 0  |  Objects: 0", font=ctk.CTkFont(size=10))
        self.lbl_perf.pack(pady=2)

        self.lbl_entities = ctk.CTkLabel(self.telemetry_frame, text="TRACKED ENTITIES:\n- None", justify="left", font=ctk.CTkFont(size=10))
        self.lbl_entities.pack(pady=2, padx=8, anchor="w")

        # Visualization Frame limits
        self.viz_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.viz_frame.pack(fill="x", expand=False)

        # Joystick Frame (Left)
        self.joystick_frame = ctk.CTkFrame(self.viz_frame)
        self.joystick_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        ctk.CTkLabel(self.joystick_frame, text="Motor Vector", font=ctk.CTkFont(size=11, weight="bold")).pack(pady=3)

        # Custom Tkinter Canvas for the Joystick Visual
        self.canvas_joy = tk.Canvas(self.joystick_frame, width=165, height=165, bg="#2b2b2b", highlightthickness=0)
        self.canvas_joy.pack(fill="both", expand=True, padx=6, pady=6)
        self.canvas_joy.bind("<Configure>", self.on_joy_resize)

        # Radar Frame (Right)
        self.radar_frame = ctk.CTkFrame(self.viz_frame)
        self.radar_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        ctk.CTkLabel(self.radar_frame, text="Predictive Path", font=ctk.CTkFont(size=11, weight="bold")).pack(pady=3)
        
        self.canvas_radar = tk.Canvas(self.radar_frame, width=165, height=165, bg="#111111", highlightthickness=0)
        self.canvas_radar.pack(fill="both", expand=True, padx=6, pady=6)
        self.canvas_radar.bind("<Configure>", self.on_radar_resize)

        self.robot_base_image = None
        self.robot_img = None
        try:
            self.robot_base_image = Image.open("robot-topdown.png").convert("RGBA")
        except Exception:
            self.robot_base_image = None

        self.lbl_motor_data = ctk.CTkLabel(self.right_panel, text="Power: 0% | Angle: 0%", font=ctk.CTkFont(size=11, weight="bold"))
        self.lbl_motor_data.pack(fill="x", pady=(3, 0))

        self.viz_frame.columnconfigure(0, weight=1)
        self.viz_frame.columnconfigure(1, weight=1)
        self.viz_frame.bind("<Configure>", lambda e: self.update_viz_layout(e.width))
        self.draw_joy_base()
        self.draw_radar_base()
        self.after(0, self.update_viz_layout)

    def update_live_output(self, frame_bgr):
        if frame_bgr is None:
            return
        if not self.winfo_exists() or not self.output_view.winfo_exists():
            return

        self.last_live_frame = frame_bgr

        # Use the containing frame's size (fixed height) instead of the label's
        # current height to avoid a feedback loop where the label's image size
        # drives its own size growth over repeated updates.
        frame_w = max(160, self.output_frame.winfo_width())
        frame_h = max(110, self.output_frame.winfo_height())
        target_w = frame_w
        target_h = frame_h

        src_h, src_w = frame_bgr.shape[:2]
        if src_w <= 0 or src_h <= 0:
            return

        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x0 = (target_w - new_w) // 2
        y0 = (target_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

        # Deadzone overlay (center band) to visualize follow turn deadzone.
        deadzone_px = max(0.0, min(float(self.follow_turn_deadzone_px), (src_w / 2.0) - 1.0))
        if deadzone_px > 0:
            dz_scaled = max(1, int(deadzone_px * scale))
            cx = target_w // 2
            left = max(0, cx - dz_scaled)
            right = min(target_w - 1, cx + dz_scaled)
            overlay = canvas.copy()
            cv2.rectangle(overlay, (left, 0), (right, target_h - 1), (0, 255, 255), -1)
            canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
            cv2.line(canvas, (left, 0), (left, target_h - 1), (0, 255, 255), 1)
            cv2.line(canvas, (right, 0), (right, target_h - 1), (0, 255, 255), 1)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.output_imgtk = ImageTk.PhotoImage(image)
        self.output_view.configure(image=self.output_imgtk)

    def on_joy_resize(self, _event=None):
        self.draw_joy_base()
        self.render_motion_visuals(self.last_fwd_mag, self.last_turn_mag)

    def on_radar_resize(self, _event=None):
        self.draw_radar_base()
        self.render_motion_visuals(self.last_fwd_mag, self.last_turn_mag)

    def draw_joy_base(self):
        w = max(80, self.canvas_joy.winfo_width())
        h = max(80, self.canvas_joy.winfo_height())
        pad = max(8, int(min(w, h) * 0.08))

        self.joy_cx = w // 2
        self.joy_cy = h // 2
        self.joy_radius = max(20, min(w, h) // 2 - pad)

        self.canvas_joy.delete("base")
        self.canvas_joy.create_oval(
            self.joy_cx - self.joy_radius,
            self.joy_cy - self.joy_radius,
            self.joy_cx + self.joy_radius,
            self.joy_cy + self.joy_radius,
            outline="#555555",
            width=2,
            tags="base",
        )
        self.canvas_joy.create_line(self.joy_cx, self.joy_cy - self.joy_radius, self.joy_cx, self.joy_cy + self.joy_radius, fill="#444444", dash=(4, 4), tags="base")
        self.canvas_joy.create_line(self.joy_cx - self.joy_radius, self.joy_cy, self.joy_cx + self.joy_radius, self.joy_cy, fill="#444444", dash=(4, 4), tags="base")

        if not self.canvas_joy.find_withtag("joy_dot"):
            self.joy_dot = self.canvas_joy.create_oval(0, 0, 0, 0, fill="cyan", outline="", tags="joy_dot")

    def draw_radar_base(self):
        w = max(80, self.canvas_radar.winfo_width())
        h = max(80, self.canvas_radar.winfo_height())
        cx = w // 2
        baseline = int(h * 0.82)

        self.radar_anchor = (cx, baseline)
        self.radar_depth_scale = max(30, int(h * 0.60))
        self.radar_turn_scale = max(20, int(w * 0.38))

        self.canvas_radar.delete("base")
        self.canvas_radar.delete("robot")
        self.canvas_radar.create_line(cx, 0, cx, h, fill="#333333", dash=(2, 4), tags="base")
        self.canvas_radar.create_line(0, baseline, w, baseline, fill="#333333", dash=(4, 4), tags="base")

        near_h = int(h * 0.25)
        far_h = int(h * 0.42)
        self.canvas_radar.create_arc(cx - int(w * 0.25), baseline - near_h, cx + int(w * 0.25), baseline + int(h * 0.10), outline="#333333", start=0, extent=180, tags="base")
        self.canvas_radar.create_arc(cx - int(w * 0.44), baseline - far_h, cx + int(w * 0.44), baseline + int(h * 0.16), outline="#333333", start=0, extent=180, tags="base")

        if self.robot_base_image is not None:
            robot_size = max(22, int(min(w, h) * 0.18))
            resized = self.robot_base_image.resize((robot_size, robot_size), Image.Resampling.LANCZOS)
            self.robot_img = ImageTk.PhotoImage(resized)
            self.canvas_radar.create_image(cx, baseline, image=self.robot_img, tags="robot")
        else:
            robot_w = max(14, int(w * 0.10))
            robot_h = max(18, int(h * 0.12))
            self.canvas_radar.create_rectangle(cx - robot_w, baseline - robot_h, cx + robot_w, baseline + robot_h, fill="gray", outline="", tags="robot")

    def update_viz_layout(self, width=None):
        if width is None:
            width = self.viz_frame.winfo_width()

        stacked = width < 320
        if getattr(self, "_viz_stacked", None) == stacked:
            return

        self._viz_stacked = stacked
        self.joystick_frame.grid_forget()
        self.radar_frame.grid_forget()

        if stacked:
            self.viz_frame.rowconfigure(0, weight=1)
            self.viz_frame.rowconfigure(1, weight=1)
            self.viz_frame.columnconfigure(0, weight=1)
            self.viz_frame.columnconfigure(1, weight=0)
            self.joystick_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
            self.radar_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        else:
            self.viz_frame.rowconfigure(0, weight=1)
            self.viz_frame.rowconfigure(1, weight=0)
            self.viz_frame.columnconfigure(0, weight=1)
            self.viz_frame.columnconfigure(1, weight=1)
            self.joystick_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
            self.radar_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

    def render_motion_visuals(self, fwd_mag, turn_mag):
        self.last_fwd_mag = fwd_mag
        self.last_turn_mag = turn_mag

        if hasattr(self, "joy_radius") and self.canvas_joy.find_withtag("joy_dot"):
            self.canvas_joy.delete("limit")
            max_speed = max(0.0, min(self.max_fwd_pct, self.max_rev_pct))
            max_turn = max(0.0, min(self.max_turn_left_pct, self.max_turn_right_pct))
            limit_rx = max(2, int(self.joy_radius * max_speed))
            limit_ry = max(2, int(self.joy_radius * max_turn))
            self.canvas_joy.create_oval(
                self.joy_cx - limit_rx,
                self.joy_cy - limit_ry,
                self.joy_cx + limit_rx,
                self.joy_cy + limit_ry,
                outline="#66aaff",
                width=2,
                dash=(4, 3),
                tags="limit",
            )
            dot_radius = max(4, int(self.joy_radius * 0.10))
            dot_x = self.joy_cx + int(fwd_mag * self.joy_radius)
            dot_y = self.joy_cy - int(turn_mag * self.joy_radius)
            self.canvas_joy.coords(self.joy_dot, dot_x - dot_radius, dot_y - dot_radius, dot_x + dot_radius, dot_y + dot_radius)
            self.canvas_joy.tag_raise("joy_dot")

        speed_pct = int(-fwd_mag * 100)  # Negative fwd_mag is Forward (+ Speed)
        dir_txt = "Fwd" if speed_pct > 0 else "Rev" if speed_pct < 0 else "Stop"
        turn_pct = int(turn_mag * 100)   # Positive turn_mag is Right
        turn_txt = "Right" if turn_pct > 0 else "Left" if turn_pct < 0 else "Straight"
        self.lbl_motor_data.configure(text=f"Power: {abs(speed_pct)}% {dir_txt}  |  Angle: {abs(turn_pct)}% {turn_txt}")

        self.canvas_radar.delete("path")
        if not hasattr(self, "radar_anchor"):
            return

        rx, ry = self.radar_anchor
        speed_px = speed_pct * (self.radar_depth_scale / 100.0)
        curve_px = turn_pct * (self.radar_turn_scale / 100.0)

        if abs(speed_px) > 2:
            pt1 = (rx, ry)
            pt2 = (int(rx + (curve_px * 0.5)), int(ry - (speed_px * 0.5)))
            pt3 = (int(rx + curve_px), int(ry - speed_px))

            path_color = "green" if speed_pct > 0 else "red"
            self.canvas_radar.create_line(
                [pt1, pt2, pt3],
                smooth=True,
                fill=path_color,
                width=3,
                arrow=tk.LAST,
                tags="path",
            )

        self.canvas_radar.tag_raise("robot")

    def update_telemetry(self, directive_text, directive_color, threat_text, perf_text, entities_text, fwd_mag, turn_mag):
        self.lbl_directive.configure(text=f"CURRENT DIRECTIVE:\n[ {directive_text} ]", text_color=directive_color)
        self.lbl_threat_matrix.configure(text=threat_text)
        self.lbl_perf.configure(text=perf_text)
        self.lbl_entities.configure(text=entities_text)
        self.render_motion_visuals(fwd_mag, turn_mag)
        self.update_looking_debug_line()

    def update_looking_debug_line(self):
        req = "ON" if self.looking_mode_requested else "OFF"
        act = "ON" if self.looking_mode else "OFF"
        if self.follow_cleanup_state != "idle":
            seq = f"exit:{self.follow_cleanup_state}"
        elif self.follow_look_state != "idle":
            seq = f"enter:{self.follow_look_state}"
        else:
            seq = "idle"

        color = "#f0c84b" if (self.looking_mode_requested or self.looking_mode) else "gray70"
        self.lbl_looking_debug.configure(text=f"dbg req:{req} act:{act} seq:{seq}", text_color=color)

    def set_active_goal(self, value):
        self.active_goal = value
        self.update_looking_debug_line()
        
    def update_target_obj(self, event=None):
        self.target_object = self.entry_target.get().lower().strip()

    def update_fwd_limit(self, value):
        limit = max(0.0, float(value) / 100.0)
        self.max_fwd_pct = limit
        self.max_rev_pct = limit
        self.fwd_sld_label.configure(text=f"Max Speed: {int(value)}%")
        self.render_motion_visuals(self.last_fwd_mag, self.last_turn_mag)

    def update_deadzone_limit(self, value):
        self.follow_turn_deadzone_px = max(0.0, float(value))
        self.deadzone_sld_label.configure(text=f"Turn Deadzone: {int(value)} px")
        if self.last_live_frame is not None:
            self.update_live_output(self.last_live_frame)

    def update_turn_limit(self, value):
        limit = max(0.0, min(1.0, float(value) / 100.0))
        self.max_turn_left_pct = limit
        self.max_turn_right_pct = limit
        self.turn_sld_label.configure(text=f"Max Turn: {int(value)}%")
        self.render_motion_visuals(self.last_fwd_mag, self.last_turn_mag)

    def toggle_looking_mode(self):
        self.looking_mode_requested = bool(self.chk_looking.get())
        self.update_looking_debug_line()

    def on_persist_mode_toggle(self):
        pass

    def set_looking_mode(self, enabled):
        enabled = bool(enabled)
        if self.looking_mode == enabled:
            return
        self.looking_mode = enabled
        self.update_looking_debug_line()

    def compute_follow_turn(self, compensated_offset_px, half_width, prev_turn_cmd, frame_scale=1.0):
        half_width = max(1.0, float(half_width))
        frame_scale = max(0.5, min(2.5, float(frame_scale)))

        deadzone_px = max(0.0, min(self.follow_turn_deadzone_px, half_width - 1.0))
        offset_abs = abs(float(compensated_offset_px))

        if offset_abs <= deadzone_px:
            desired_turn = 0.0
        else:
            usable = max(1.0, half_width - deadzone_px)
            norm = min(1.0, (offset_abs - deadzone_px) / usable)
            shaped = norm ** 1.15
            desired_turn = float(np.sign(compensated_offset_px)) * min(1.0, shaped * self.follow_turn_gain)
            if abs(desired_turn) < self.follow_turn_min_cmd:
                desired_turn = float(np.sign(compensated_offset_px)) * self.follow_turn_min_cmd

        max_step = self.follow_turn_rate_limit * frame_scale
        low = prev_turn_cmd - max_step
        high = prev_turn_cmd + max_step
        return max(-1.0, min(1.0, max(low, min(high, desired_turn))))

    def set_rgb_region(self):
        self.withdraw()
        
        def on_selected(region):
            self.rgb_monitor = region
            self.rgb_region_label.configure(text=f"RGB Zone: {self.rgb_monitor['width']}x{self.rgb_monitor['height']} at ({self.rgb_monitor['left']}, {self.rgb_monitor['top']})")
            
        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    def set_depth_region(self):
        self.withdraw()
        
        def on_selected(region):
            self.depth_monitor = region
            self.depth_region_label.configure(text=f"Depth Zone: {self.depth_monitor['width']}x{self.depth_monitor['height']} at ({self.depth_monitor['left']}, {self.depth_monitor['top']})")
            
        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    def set_controller_region(self):
        self.withdraw()
        
        def on_selected(region):
            self.controller_monitor = region
            self.btn_set_controller.configure(text=f"Output Zone: {self.controller_monitor['width']}x{self.controller_monitor['height']} at ({self.controller_monitor['left']}, {self.controller_monitor['top']})")
            if self.translate_motion:
                self.mouse_hold_arm_time = time.time() + self.mouse_hold_delay_sec
                self.set_mouse_hold(False)
            
        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    @staticmethod
    def _region_center(region):
        return (
            int(region["left"] + (region["width"] // 2)),
            int(region["top"] + (region["height"] // 2)),
        )

    @staticmethod
    def _spot_to_list(spot):
        if spot is None:
            return None
        return [int(spot[0]), int(spot[1])]

    @staticmethod
    def _list_to_spot(value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None

    def get_spots_file_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_spots.json")

    def update_spot_labels(self):
        self.lbl_uv_spot.configure(
            text=f"UV Spot: ({self.uv_spot[0]}, {self.uv_spot[1]})" if self.uv_spot else "UV Spot: Not Set"
        )
        self.lbl_tripod_spot.configure(
            text=f"Tripod Spot: ({self.tripod_spot[0]}, {self.tripod_spot[1]})" if self.tripod_spot else "Tripod Spot: Not Set"
        )
        self.lbl_hold_pos_spot.configure(
            text=f"Hold Position Spot: ({self.hold_position_spot[0]}, {self.hold_position_spot[1]})" if self.hold_position_spot else "Hold Position: Not Set"
        )
        self.lbl_rear_spot.configure(
            text=f"Rear-most Spot: ({self.rear_most_spot[0]}, {self.rear_most_spot[1]})" if self.rear_most_spot else "Rear-most: Not Set"
        )
        self.lbl_zw_spot.configure(
            text=f"ZW Spot: ({self.zw_spot[0]}, {self.zw_spot[1]})" if self.zw_spot else "ZW Spot: Not Set"
        )
        self.lbl_focus_spot.configure(
            text=f"Focus Window: ({self.focus_window_spot[0]}, {self.focus_window_spot[1]})" if self.focus_window_spot else "Focus Window: Not Set"
        )

    def save_spots(self):
        payload = {
            "uv_spot": self._spot_to_list(self.uv_spot),
            "tripod_spot": self._spot_to_list(self.tripod_spot),
            "hold_position_spot": self._spot_to_list(self.hold_position_spot),
            "rear_most_spot": self._spot_to_list(self.rear_most_spot),
            "zw_spot": self._spot_to_list(self.zw_spot),
            "focus_window_spot": self._spot_to_list(self.focus_window_spot),
        }

        try:
            with open(self.get_spots_file_path(), "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            self.label.configure(text="Hex-Vision Object Tracker (spots saved)")
        except Exception as exc:
            messagebox.showerror("Save Spots", f"Could not save spots:\n{exc}")

    def load_saved_spots(self):
        path = self.get_spots_file_path()
        if not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.uv_spot = self._list_to_spot(payload.get("uv_spot"))
            self.tripod_spot = self._list_to_spot(payload.get("tripod_spot"))
            self.hold_position_spot = self._list_to_spot(payload.get("hold_position_spot"))
            self.rear_most_spot = self._list_to_spot(payload.get("rear_most_spot"))
            self.zw_spot = self._list_to_spot(payload.get("zw_spot"))
            self.focus_window_spot = self._list_to_spot(payload.get("focus_window_spot"))

            self.update_spot_labels()
            self.label.configure(text="Hex-Vision Object Tracker (spots loaded)")
        except Exception as exc:
            messagebox.showerror("Load Spots", f"Could not load saved spots:\n{exc}")

    def set_uv_spot(self):
        self.withdraw()

        def on_selected(region):
            self.uv_spot = self._region_center(region)
            self.update_spot_labels()

        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    def set_tripod_spot(self):
        self.withdraw()

        def on_selected(region):
            self.tripod_spot = self._region_center(region)
            self.update_spot_labels()

        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    def set_hold_position_spot(self):
        self.withdraw()

        def on_selected(region):
            self.hold_position_spot = self._region_center(region)
            self.update_spot_labels()

        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    def set_rear_most_spot(self):
        self.withdraw()

        def on_selected(region):
            self.rear_most_spot = self._region_center(region)
            self.update_spot_labels()

        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    def set_zw_spot(self):
        self.withdraw()

        def on_selected(region):
            self.zw_spot = self._region_center(region)
            self.update_spot_labels()

        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    def set_focus_window_spot(self):
        self.withdraw()

        def on_selected(region):
            self.focus_window_spot = self._region_center(region)
            self.update_spot_labels()

        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    def run_follow_enter_sequence(self):
        if self.follow_look_state == "idle":
            if self.focus_window_spot is None:
                return False, "SET FOCUS WINDOW SPOT", (0, 165, 255)
            if self.click_screen_point(self.focus_window_spot):
                self.follow_look_state = "enable_hold"
                return False, "FOCUSING WINDOW", (0, 165, 255)
            return False, "FOCUSING WINDOW", (0, 165, 255)

        steps = [
            ("enable_hold", self.hold_position_spot, "SET HOLD POSITION SPOT", "ENABLING HOLD POSITION"),
            ("enter_uv", self.uv_spot, "SET UV SPOT", "ENTERING UV MODE"),
            ("set_rear", self.rear_most_spot, "SET REAR-MOST SPOT", "SLIDING TO REAR POSITION"),
            ("enter_zw", self.zw_spot, "SET ZW SPOT", "ENTERING ZW MODE"),
        ]

        for idx, (state_name, spot, missing_text, progress_text) in enumerate(steps):
            if self.follow_look_state != state_name:
                continue

            if spot is None:
                return False, missing_text, (0, 165, 255)

            if self.click_screen_point(spot):
                if idx == len(steps) - 1:
                    self.follow_look_state = "active"
                    self.follow_cleanup_state = "idle"
                    self.set_looking_mode(True)
                    return True, "ZW LOOK ACTIVE", (255, 0, 255)
                self.follow_look_state = steps[idx + 1][0]
            return False, progress_text, (255, 0, 255)

        if self.follow_look_state == "active":
            self.set_looking_mode(True)
            return True, "ZW LOOK ACTIVE", (255, 0, 255)

        return False, "PREPARING LOOK SEQUENCE", (255, 0, 255)

    def run_follow_exit_sequence(self):
        if self.follow_look_state == "idle" and self.follow_cleanup_state == "idle":
            return True, "FOLLOW MODE", (0, 255, 0)

        if self.follow_cleanup_state == "idle":
            if self.focus_window_spot is None:
                return False, "SET FOCUS WINDOW SPOT", (0, 165, 255)
            if self.click_screen_point(self.focus_window_spot):
                self.follow_cleanup_state = "disable_hold"
                return False, "FOCUSING WINDOW", (0, 165, 255)
            return False, "FOCUSING WINDOW", (0, 165, 255)

        if self.follow_cleanup_state == "disable_hold":
            if self.hold_position_spot is None:
                return False, "SET HOLD POSITION SPOT", (0, 165, 255)
            if self.click_screen_point(self.hold_position_spot):
                self.follow_cleanup_state = "tripod"
            self.set_looking_mode(False)
            return False, "DISABLING HOLD POSITION", (0, 165, 255)

        if self.follow_cleanup_state == "tripod":
            if self.tripod_spot is None:
                return False, "SET TRIPOD SPOT", (0, 165, 255)
            if self.click_screen_point(self.tripod_spot):
                self.follow_cleanup_state = "idle"
                self.follow_look_state = "idle"
                self.set_looking_mode(False)
                return True, "RETURNED TO TRIPOD", (0, 255, 0)
            return False, "RETURNING TO TRIPOD", (0, 165, 255)

        self.follow_cleanup_state = "idle"
        self.follow_look_state = "idle"
        self.set_looking_mode(False)
        return True, "FOLLOW MODE", (0, 255, 0)

    def click_screen_point(self, point):
        if point is None:
            return False

        now = time.time()
        if (now - self.last_mode_click_time) < self.mode_click_cooldown_sec:
            return False

        try:
            import ctypes

            was_holding = self.mouse_left_is_down
            if was_holding:
                self.set_mouse_hold(False)

            ctypes.windll.user32.SetCursorPos(int(point[0]), int(point[1]))
            ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
            ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)

            if self.translate_motion and was_holding:
                self.mouse_hold_arm_time = time.time() + self.mouse_hold_delay_sec

            self.last_mode_click_time = now
            return True
        except Exception:
            return False

    def set_mouse_hold(self, should_hold):
        if should_hold == self.mouse_left_is_down:
            return
        try:
            import ctypes
            flag = 0x0002 if should_hold else 0x0004
            ctypes.windll.user32.mouse_event(flag, 0, 0, 0, 0)
            self.mouse_left_is_down = should_hold
        except Exception:
            self.mouse_left_is_down = False

    def update_autonomy_visual(self):
        enabled = bool(self.autonomy_enabled.get())
        if enabled:
            self.autonomy_btn.configure(
                text="Autonomous Input: ON",
                fg_color="#1f6f3d",
                hover_color="#2a8a4b",
                text_color="white",
                border_width=2,
                border_color="#33c26b",
            )
        else:
            self.autonomy_btn.configure(
                text="Autonomous Input: OFF",
                fg_color="#2b2f36",
                hover_color="#3a3f48",
                text_color="#e0e0e0",
                border_width=2,
                border_color="#5a5f68",
            )

    def toggle_controller(self, force=None):
        if force is None:
            enabled = not self.autonomy_enabled.get()
        else:
            enabled = bool(force)
        self.autonomy_enabled.set(enabled)
        self.translate_motion = enabled
        self.update_autonomy_visual()
        if self.translate_motion:
            self.mouse_hold_arm_time = time.time() + self.mouse_hold_delay_sec
            self.set_mouse_hold(False)
        else:
            self.mouse_hold_arm_time = None
            self.set_mouse_hold(False)

    def load_model(self):
        if self.model is None:
            self.label.configure(text="Loading Model...")
            self.update()
            # Keep the local model file but run it in person-only, box-driven mode for speed.
            self.model = YOLO("yolov8n-seg.pt")
            try:
                self.model.fuse()
            except Exception:
                pass
            self.label.configure(text="Hex-Vision Object Tracker")

    def start_vision(self):
        self.load_model()
        self.is_running = True
        self.follow_look_state = "idle"
        self.follow_cleanup_state = "idle"
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_set_rgb.configure(state="disabled")
        self.btn_set_depth.configure(state="disabled")

        self.capture_thread = threading.Thread(target=self.vision_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def stop_vision(self):
        self.is_running = False
        self.follow_look_state = "idle"
        self.follow_cleanup_state = "idle"
        self.set_looking_mode(False)
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_set_rgb.configure(state="normal")
        self.btn_set_depth.configure(state="normal")
        self.mouse_hold_arm_time = None
        self.set_mouse_hold(False)

    def vision_loop(self):
        sct = mss.mss()
        prev_time = time.time()
        
        # Smoothing variables for motor control
        smoothed_fwd = 0.0
        smoothed_turn = 0.0

        # Follow persistence and turn-shaping state.
        last_follow_seen_time = 0.0
        persisted_target_offset_px = 0.0
        persisted_follow_fwd = 0.0
        prev_follow_turn_cmd = 0.0
        last_seen_turn_sign = 0.0
        
        # Load ctypes for global Escape monitor
        try:
            import ctypes
            user32 = ctypes.windll.user32
        except ImportError:
            user32 = None
        esc_was_down = False

        while self.is_running:
            persist_turn_boost_active = False
            # Global Escape disables mouse translation without exiting the app.
            esc_down = bool(user32 and (user32.GetAsyncKeyState(0x1B) & 0x8000))
            if esc_down and not esc_was_down:
                if self.translate_motion:
                    print("Escape detected: disabling mouse translation.")
                    self.toggle_controller(False)
            esc_was_down = esc_down
                
            # Measure FPS
            curr_time = time.time()
            frame_dt = curr_time - prev_time
            if frame_dt <= 0:
                frame_dt = 1.0 / 60.0
            fps = 1.0 / frame_dt
            prev_time = curr_time
            frame_scale = max(0.5, min(2.0, frame_dt * 30.0))
            
            detected_objects = []
            person_boxes = []
            
            # Capture RGB screen
            rgb_screenshot = sct.grab(self.rgb_monitor)
            frame_bgra = np.array(rgb_screenshot, dtype=np.uint8)
            frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            
            # Capture Depth screen if configured
            depth_frame = None
            if self.depth_monitor:
                depth_screenshot = sct.grab(self.depth_monitor)
                depth_frame_bgra = np.array(depth_screenshot, dtype=np.uint8)
                # Convert Depth to grayscale using RED channel for screen capture.
                depth_frame = depth_frame_bgra[:, :, 2]

            # Resize depth exactly to match RGB width and height
            if depth_frame is not None and depth_frame.shape[:2] != frame.shape[:2]:
                depth_frame = cv2.resize(depth_frame, (frame.shape[1], frame.shape[0]))

            target_cx = None
            target_depth = None
            target_cy = None
            target_best_key = None

            # Always run person-only inference so live output consistently shows boxes.
            results = self.model(
                frame,
                verbose=False,
                classes=[0],
                conf=self.infer_conf,
                iou=self.infer_iou,
                imgsz=self.infer_imgsz,
                max_det=self.infer_max_det,
            )

            frame_h, frame_w = frame.shape[:2]
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                boxes_conf = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones(len(boxes_xyxy), dtype=np.float32)

                for idx, box in enumerate(boxes_xyxy):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1 = max(0, min(frame_w - 1, x1))
                    y1 = max(0, min(frame_h - 1, y1))
                    x2 = max(0, min(frame_w - 1, x2))
                    y2 = max(0, min(frame_h - 1, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    cX = (x1 + x2) // 2
                    cY = (y1 + y2) // 2
                    conf_pct = int(float(boxes_conf[idx]) * 100)

                    mean_depth = 0.0
                    distance_text = ""
                    box_color = (255, 255, 255)

                    if depth_frame is not None:
                        roi = depth_frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            close_pixels = roi[roi > 80]
                            mean_depth = float(np.mean(close_pixels)) if close_pixels.size > 0 else 0.0

                        if mean_depth > 220:
                            distance_text = "CRITICAL"
                            box_color = (0, 0, 255)
                        elif mean_depth > 175:
                            distance_text = "SEVERE"
                            box_color = (0, 69, 255)
                        elif mean_depth > 130:
                            distance_text = "APPROACHING"
                            box_color = (0, 140, 255)
                        elif mean_depth > 80:
                            distance_text = "NEAR"
                            box_color = (0, 255, 255)
                        else:
                            distance_text = "SAFE"
                            box_color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    label_text = f"person {conf_pct}%"
                    if distance_text:
                        label_text += f" {distance_text}"
                    cv2.putText(
                        frame,
                        label_text,
                        (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2,
                        cv2.LINE_AA,
                    )

                    detected_objects.append(label_text)
                    person_boxes.append((x1, y1, x2, y2, box_color))

                    # Prioritize the closest person: depth when available, bbox area otherwise.
                    center_error = abs(cX - (frame_w * 0.5))
                    box_area = float((x2 - x1) * (y2 - y1))
                    has_depth_signal = 1 if mean_depth > 0 else 0
                    proximity_score = float(mean_depth) if has_depth_signal else box_area
                    candidate_key = (has_depth_signal, proximity_score, -center_error)

                    if target_best_key is None or candidate_key > target_best_key:
                        target_best_key = candidate_key
                        target_cx = cX
                        target_cy = cY
                        target_depth = mean_depth

            if target_cx is not None and target_cy is not None:
                cv2.circle(frame, (target_cx, target_cy), 5, (255, 0, 255), -1)

            if person_boxes:
                aim_x = int(target_cx) if target_cx is not None else (frame_w // 2)
                for x1, y1, x2, y2, box_color in person_boxes:
                    line_x = max(x1, min(x2, aim_x))
                    cv2.line(frame, (line_x, y1), (line_x, y2), box_color, 2)

            # === ROBOT BRAIN (LIVE COMMAND INTERPRETER) ===
            height_f, width_f = frame.shape[:2]
            third_w = width_f // 3

            # Draw grid lines for visualizing robot view columns back onto main preview
            cv2.line(frame, (third_w, 0), (third_w, height_f), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (2 * third_w, 0), (2 * third_w, height_f), (255, 255, 255), 1, cv2.LINE_AA)
            
            action = "MOVE FORWARD"
            action_color = (0, 255, 0)
            
            l_threat, c_threat, r_threat = 0, 0, 0
            
            if depth_frame is not None:
                # We analyze intensity of depth pixels (nearness) in Left, Center, Right cols
                # Filter for pixels that are > 80 (meaning they are somewhat present/not infinite background)
                l_roi = depth_frame[:, :third_w]
                c_roi = depth_frame[:, third_w:2*third_w]
                r_roi = depth_frame[:, 2*third_w:]
                
                # Check mean density of obstacles in each grid section
                l_threat = np.mean(l_roi[l_roi > 80]) if np.any(l_roi > 80) else 0
                c_threat = np.mean(c_roi[c_roi > 80]) if np.any(c_roi > 80) else 0
                r_threat = np.mean(r_roi[r_roi > 80]) if np.any(r_roi > 80) else 0

                # Threat values range up to ~255 based on proximity
                if self.active_goal != "Follow Object" and self.looking_mode_requested:
                    fwd_mag = 0.0
                    turn_mag = 0.0
                    seq_ready, seq_text, seq_color = self.run_follow_enter_sequence()
                    action = f"LOOK DEBUG: {seq_text}"
                    action_color = seq_color
                elif self.active_goal != "Follow Object" and (self.follow_look_state != "idle" or self.follow_cleanup_state != "idle"):
                    fwd_mag = 0.0
                    turn_mag = 0.0
                    exit_done, exit_text, exit_color = self.run_follow_exit_sequence()
                    action = f"LOOK DEBUG: {exit_text}"
                    action_color = exit_color
                elif self.active_goal == "Follow Object" and self.target_object:
                    center_x = width_f / 2
                    half_width = width_f / 2

                    if target_cx is not None:
                        # We see the target!
                        action = f"FOLLOWING: {self.target_object.upper()}"
                        action_color = (0, 255, 0)
                        
                        # Determine turn based on horizontal position
                        raw_offset = target_cx - center_x

                        # Compensate apparent target drift introduced by our own turn command.
                        camera_offset_comp_px = smoothed_turn * half_width * self.turn_comp_gain
                        compensated_offset = raw_offset + camera_offset_comp_px
                        turn_mag = self.compute_follow_turn(compensated_offset, half_width, prev_follow_turn_cmd, frame_scale)
                        last_follow_seen_time = curr_time
                        persisted_target_offset_px = raw_offset
                        last_seen_turn_sign = 1.0 if raw_offset >= 0 else -1.0
                        
                        want_looking_mode = self.looking_mode_requested

                        if want_looking_mode:
                            # Looking mode: run hold->UV->rear->ZW sequence, then look only with turn.
                            base_fwd_mag = 0.0
                            seq_ready, seq_text, seq_color = self.run_follow_enter_sequence()
                            action = f"{seq_text}: {self.target_object.upper()}"
                            action_color = seq_color
                            if not seq_ready:
                                turn_mag = 0.0
                        else:
                            # Looking mode is not requested: exit sequence (hold off -> tripod), then normal follow.
                            exit_done, exit_text, exit_color = self.run_follow_exit_sequence()
                            if not exit_done:
                                base_fwd_mag = 0.0
                                action = f"{exit_text}: {self.target_object.upper()}"
                                action_color = exit_color
                                turn_mag = 0.0
                            else:
                                # Determine fwd based on depth (we want it to be ~130 distance)
                                if target_depth is not None:
                                    if target_depth > 180:
                                        # Too close, reverse!
                                        base_fwd_mag = 1.0 # Reverse
                                        action = f"BACKING UP FROM: {self.target_object.upper()}"
                                        action_color = (0, 0, 255)
                                    elif target_depth > 130:
                                        # Good distance, stop
                                        base_fwd_mag = 0.0
                                        action = f"HOLDING AT: {self.target_object.upper()}"
                                        action_color = (0, 255, 255)
                                    else:
                                        # Move forward to follow
                                        base_fwd_mag = -1.0 # Forward
                                else:
                                    base_fwd_mag = 0.0

                        fwd_mag = base_fwd_mag
                        prev_follow_turn_cmd = turn_mag
                        persisted_follow_fwd = fwd_mag
                    else:
                        time_since_seen = curr_time - last_follow_seen_time if last_follow_seen_time > 0 else 999.0
                        can_persist = (
                            time_since_seen <= self.follow_persist_sec
                            and not self.looking_mode_requested
                            and self.follow_look_state == "idle"
                            and self.follow_cleanup_state == "idle"
                        )

                        if can_persist:
                            # Predict image-space drift of a stationary world target from our own turn.
                            persisted_target_offset_px -= smoothed_turn * half_width * self.turn_comp_gain * frame_scale
                            persisted_target_offset_px = max(-half_width, min(half_width, persisted_target_offset_px))

                            predicted_comp_offset = persisted_target_offset_px + (smoothed_turn * half_width * self.turn_comp_gain)
                            if last_seen_turn_sign != 0.0:
                                turn_mag = self.follow_persist_turn_boost if last_seen_turn_sign > 0 else -self.follow_persist_turn_boost
                            else:
                                turn_mag = self.follow_persist_turn_boost if predicted_comp_offset >= 0 else -self.follow_persist_turn_boost
                            persist_turn_boost_active = True
                            fwd_mag = 0.0
                            prev_follow_turn_cmd = turn_mag

                            action = f"PERSISTING (TURN ONLY): {self.target_object.upper()}"
                            action_color = (0, 255, 255)
                        else:
                            action = f"WAITING FOR: {self.target_object.upper()}"
                            action_color = (150, 150, 150)
                            fwd_mag = 0.0
                            turn_mag = 0.0
                            prev_follow_turn_cmd = 0.0

                        if self.looking_mode_requested:
                            seq_ready, seq_text, seq_color = self.run_follow_enter_sequence()
                            action = f"{seq_text}: {self.target_object.upper()}"
                            action_color = seq_color
                        else:
                            exit_done, exit_text, exit_color = self.run_follow_exit_sequence()
                            if not exit_done:
                                action = f"{exit_text}: {self.target_object.upper()}"
                                action_color = exit_color
                        
                elif self.active_goal == "Search Space" and self.target_object and target_cx is None:
                    # We are in search mode and haven't found it yet
                    action = f"PATROLLING FOR: {self.target_object.upper()}"
                    action_color = (255, 0, 255)
                    # Gentle forward and turn to explore
                    fwd_mag = -0.5
                    turn_mag = 0.4
                    # Basic obstacle avoidance while searching
                    if c_threat > 160:
                        fwd_mag = 0.0
                        turn_mag = -0.8
                elif self.active_goal == "Search Space" and self.target_object and target_cx is not None:
                    action = f"FOUND: {self.target_object.upper()}"
                    action_color = (0, 255, 0)
                    fwd_mag = 0.0 # Stop!
                    turn_mag = 0.0
                    
                else: 
                    prev_follow_turn_cmd = 0.0
                    # Default: Avoid All Objects
                    if c_threat > 180 and l_threat > 180 and r_threat > 180:
                        action = "BACK UP / REVERSE"
                        action_color = (0, 0, 255)
                        fwd_mag = 1.0  # Max Reverse (Rightmost)
                        turn_mag = 0.0 # Straight
                    elif c_threat > 140:
                        # Center blocked, figure out which side is safer
                        fwd_mag = 0.0 # Stopping to turn
                        if l_threat < r_threat:
                            action = "TURN LEFT"
                            action_color = (0, 165, 255)
                            turn_mag = -1.0 # Max Left (Down)
                        else:
                            action = "TURN RIGHT"
                            action_color = (0, 165, 255)
                            turn_mag = 1.0  # Max Right (Up)
                    elif l_threat > 160:
                        action = "VEER RIGHT"
                        action_color = (0, 255, 255)
                        fwd_mag = -0.5  # Moving Forward moderately
                        turn_mag = 0.8  # Strong Right (Up)
                    elif r_threat > 160:
                        action = "VEER LEFT"
                        action_color = (0, 255, 255)
                        fwd_mag = -0.5  # Moving Forward moderately
                        turn_mag = -0.8 # Strong Left (Down)
                    else:
                        action = "ALL CLEAR -> MOVE FORWARD"
                        action_color = (0, 255, 0)
                        
                        # Compute continuous forward acceleration based on how clear it is
                        threat_val = max(0, min(140, c_threat))
                        fwd_mag = -1.0 + (threat_val / 200.0)
                        
                        # Continuous turn balancing (Micro-corrections left/right)
                        diff = l_threat - r_threat
                        turn_mag = diff / 150.0  # positive = turn right (Up)
                        turn_mag = max(-1.0, min(1.0, turn_mag))
                        
                # Apply Throttle Clamping
                if fwd_mag < 0:
                    # Going forward, limit by max_fwd_pct
                    fwd_mag = max(fwd_mag, -self.max_fwd_pct)
                elif fwd_mag > 0:
                    # Going reverse, limit by max_rev_pct
                    fwd_mag = min(fwd_mag, self.max_rev_pct)
                    
                # Display metrics at top grid intersections on main preview
                cv2.putText(frame, f"Threat: {int(l_threat)}", (third_w // 2 - 40, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
                cv2.putText(frame, f"Threat: {int(c_threat)}", (third_w + (third_w // 2) - 40, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
                cv2.putText(frame, f"Threat: {int(r_threat)}", ((2 * third_w) + (third_w // 2) - 40, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
            else:
                fwd_mag = 0.0
                turn_mag = 0.0
                if self.active_goal != "Follow Object" and self.looking_mode_requested:
                    seq_ready, seq_text, seq_color = self.run_follow_enter_sequence()
                    action = f"LOOK DEBUG: {seq_text}"
                    action_color = seq_color
                elif self.active_goal != "Follow Object" and (self.follow_look_state != "idle" or self.follow_cleanup_state != "idle"):
                    exit_done, exit_text, exit_color = self.run_follow_exit_sequence()
                    action = f"LOOK DEBUG: {exit_text}"
                    action_color = exit_color
                else:
                    action = "WAITING FOR DEPTH FEED"
                    action_color = (150, 150, 150)
                    prev_follow_turn_cmd = 0.0

            # Global turn clamp: apply asymmetric max left/right steering limits.
            if persist_turn_boost_active:
                turn_mag = max(-self.follow_persist_turn_boost, min(self.follow_persist_turn_boost, turn_mag))
            else:
                turn_mag = max(-self.max_turn_left_pct, min(self.max_turn_right_pct, turn_mag))

            # Update the Tkinter Telemetry Dashboard safely
            b_color_tk = "green"
            if action_color == (0, 0, 255): b_color_tk = "red"
            elif action_color == (0, 165, 255): b_color_tk = "orange"
            elif action_color == (0, 255, 255): b_color_tk = "yellow"
            elif action_color == (150, 150, 150): b_color_tk = "gray"
            
            threat_text = f"THREAT MATRIX\nL: {int(l_threat)}   |   C: {int(c_threat)}   |   R: {int(r_threat)}"
            perf_text = f"FPS: {fps:.1f}   |   Objects: {len(detected_objects)}"
            
            ent_list = "\n".join([f" - {d}" for d in detected_objects[:6]])
            if not ent_list: ent_list = " - None"
            elif len(detected_objects) > 6: ent_list += f"\n   ...and {len(detected_objects)-6} more"
            entities_text = f"TRACKED ENTITIES:\n{ent_list}"
            
            # Apply EMA smoothing with higher responsiveness to cut control lag.
            alpha = 0.20  # 20% new reactivity, 80% history
            smoothed_fwd = (alpha * fwd_mag) + ((1.0 - alpha) * smoothed_fwd)
            smoothed_turn = (alpha * turn_mag) + ((1.0 - alpha) * smoothed_turn)
            
            # push to UI
            self.after(0, self.update_telemetry, action, b_color_tk, threat_text, perf_text, entities_text, smoothed_fwd, smoothed_turn)
            self.after(0, self.update_live_output, frame.copy())
            
            # Apply motion vector to the mouse controller if enabled
            if self.translate_motion and self.controller_monitor:
                import ctypes
                
                # map exactly like the joystick canvas
                region_cx = self.controller_monitor['left'] + (self.controller_monitor['width'] // 2)
                region_cy = self.controller_monitor['top'] + (self.controller_monitor['height'] // 2)
                
                # circle fits entirely within the region
                radius = min(self.controller_monitor['width'], self.controller_monitor['height']) // 2
                
                mouse_x = int(region_cx + (smoothed_fwd * radius))
                mouse_y = int(region_cy - (smoothed_turn * radius))
                
                # Move cursor
                ctypes.windll.user32.SetCursorPos(mouse_x, mouse_y)

                if self.mouse_hold_arm_time is not None and time.time() >= self.mouse_hold_arm_time:
                    self.set_mouse_hold(True)
                    self.mouse_hold_arm_time = None
            else:
                self.set_mouse_hold(False)

        # Safely destroy windows when loop terminates
        self.mouse_hold_arm_time = None
        self.set_mouse_hold(False)

if __name__ == "__main__":
    app = HexVisionApp()
    app.mainloop()
