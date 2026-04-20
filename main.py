import cv2
import numpy as np
import mss
import time
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox
import customtkinter as ctk
from ultralytics import YOLO
from PIL import Image, ImageTk, ImageDraw, ImageFont

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
        self.title("Hex-Vision")
        self.geometry("600x1000")
        self.resizable(True, True)
        self.base_window_width = 600
        self.base_window_height = 1000
        self._ui_scale = 1.0
        self._viz_stacked = None
        self.last_fwd_mag = 0.0
        self.last_turn_mag = 0.0
        self.mouse_hold_delay_sec = 1.0
        self.mouse_hold_arm_time = None
        self.mouse_left_is_down = False
        self.mode_click_cooldown_sec = 0.6
        self.last_mode_click_time = 0.0
        self.turn_comp_gain = 0.55

        # state
        self.is_running = False
        self.capture_thread = None
        
        # Robot Goal states
        self.active_goal = "Avoid All Objects"
        self.target_object = "person"
        self.max_fwd_pct = 1.0
        self.max_rev_pct = 1.0
        self.looking_mode = False
        self.looking_mode_requested = False
        self.super_close_threshold = 220.0
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
        self.fit_window_to_screen()
        self.bind("<Configure>", self.on_window_resize)

    def fit_window_to_screen(self):
        screen_w = max(1, self.winfo_screenwidth())
        screen_h = max(1, self.winfo_screenheight())

        # Keep margins so taskbar/title bar do not clip content on smaller displays.
        max_w = max(480, min(screen_w - 80, int(screen_w * 0.96)))
        max_h = max(520, min(screen_h - 120, int(screen_h * 0.92)))

        start_w = min(self.base_window_width, max_w)
        start_h = min(self.base_window_height, max_h)

        pos_x = max(0, (screen_w - start_w) // 2)
        pos_y = max(0, (screen_h - start_h) // 2)

        self.maxsize(max_w, max_h)
        self.geometry(f"{start_w}x{start_h}+{pos_x}+{pos_y}")

        scale = min(start_w / self.base_window_width, start_h / self.base_window_height)
        scale = max(0.50, min(1.35, scale))
        self._ui_scale = scale
        ctk.set_widget_scaling(scale)

    def setup_ui(self):
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(4, weight=1)
        self.main_container.grid_rowconfigure(5, weight=1)

        self.label = ctk.CTkLabel(self.main_container, text="Hex-Vision Object Tracker", font=ctk.CTkFont(size=20, weight="bold"))
        self.label.grid(row=0, column=0, padx=20, pady=(8, 16), sticky="ew")

        # Region Frame
        self.region_frame = ctk.CTkFrame(self.main_container)
        self.region_frame.grid(row=1, column=0, padx=20, pady=5, sticky="ew")

        self.rgb_region_label = ctk.CTkLabel(self.region_frame, text=f"RGB Zone: {self.rgb_monitor['width']}x{self.rgb_monitor['height']} at ({self.rgb_monitor['left']}, {self.rgb_monitor['top']})")
        self.rgb_region_label.pack(pady=5)

        self.btn_set_rgb = ctk.CTkButton(self.region_frame, text="Set RGB Camera Zone", command=self.set_rgb_region)
        self.btn_set_rgb.pack(pady=5)

        self.depth_region_label = ctk.CTkLabel(self.region_frame, text="Depth Zone: Not Set")
        self.depth_region_label.pack(pady=5)

        self.btn_set_depth = ctk.CTkButton(self.region_frame, text="Set Depth Camera Zone", command=self.set_depth_region)
        self.btn_set_depth.pack(pady=5)
        
        # Controller Mapping
        self.chk_controller = ctk.CTkCheckBox(self.region_frame, text="Translate Motion Vector to Mouse", command=self.toggle_controller)
        self.chk_controller.pack(pady=5)
        
        self.btn_set_controller = ctk.CTkButton(self.region_frame, text="Set Controller Output Region", command=self.set_controller_region)
        self.btn_set_controller.pack(pady=5)

        self.spot_table = ctk.CTkFrame(self.region_frame)
        self.spot_table.pack(fill="x", padx=8, pady=(8, 4))
        self.spot_table.columnconfigure(0, weight=1)
        self.spot_table.columnconfigure(1, weight=0)
        self.spot_table.columnconfigure(2, weight=1)
        self.spot_table.columnconfigure(3, weight=0)

        ctk.CTkLabel(self.spot_table, text="Spot Setup", font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=0, columnspan=4, pady=(4, 6), sticky="w")

        self.lbl_uv_spot = ctk.CTkLabel(self.spot_table, text="UV Spot: Not Set", anchor="w")
        self.lbl_uv_spot.grid(row=1, column=0, padx=(6, 6), pady=2, sticky="ew")
        self.btn_set_uv_spot = ctk.CTkButton(self.spot_table, text="Set", width=70, command=self.set_uv_spot)
        self.btn_set_uv_spot.grid(row=1, column=1, padx=(0, 10), pady=2)

        self.lbl_tripod_spot = ctk.CTkLabel(self.spot_table, text="Tripod Spot: Not Set", anchor="w")
        self.lbl_tripod_spot.grid(row=1, column=2, padx=(6, 6), pady=2, sticky="ew")
        self.btn_set_tripod_spot = ctk.CTkButton(self.spot_table, text="Set", width=70, command=self.set_tripod_spot)
        self.btn_set_tripod_spot.grid(row=1, column=3, padx=(0, 6), pady=2)

        self.lbl_hold_pos_spot = ctk.CTkLabel(self.spot_table, text="Hold Position: Not Set", anchor="w")
        self.lbl_hold_pos_spot.grid(row=2, column=0, padx=(6, 6), pady=2, sticky="ew")
        self.btn_set_hold_pos_spot = ctk.CTkButton(self.spot_table, text="Set", width=70, command=self.set_hold_position_spot)
        self.btn_set_hold_pos_spot.grid(row=2, column=1, padx=(0, 10), pady=2)

        self.lbl_rear_spot = ctk.CTkLabel(self.spot_table, text="Rear-most: Not Set", anchor="w")
        self.lbl_rear_spot.grid(row=2, column=2, padx=(6, 6), pady=2, sticky="ew")
        self.btn_set_rear_spot = ctk.CTkButton(self.spot_table, text="Set", width=70, command=self.set_rear_most_spot)
        self.btn_set_rear_spot.grid(row=2, column=3, padx=(0, 6), pady=2)

        self.lbl_zw_spot = ctk.CTkLabel(self.spot_table, text="ZW Spot: Not Set", anchor="w")
        self.lbl_zw_spot.grid(row=3, column=0, padx=(6, 6), pady=(2, 6), sticky="ew")
        self.btn_set_zw_spot = ctk.CTkButton(self.spot_table, text="Set", width=70, command=self.set_zw_spot)
        self.btn_set_zw_spot.grid(row=3, column=1, padx=(0, 10), pady=(2, 6))

        self.lbl_focus_spot = ctk.CTkLabel(self.spot_table, text="Focus Window: Not Set", anchor="w")
        self.lbl_focus_spot.grid(row=3, column=2, padx=(6, 6), pady=(2, 6), sticky="ew")
        self.btn_set_focus_spot = ctk.CTkButton(self.spot_table, text="Set", width=70, command=self.set_focus_window_spot)
        self.btn_set_focus_spot.grid(row=3, column=3, padx=(0, 6), pady=(2, 6))

        # Control Frame
        self.control_frame = ctk.CTkFrame(self.main_container)
        self.control_frame.grid(row=2, column=0, padx=20, pady=5, sticky="ew")

        self.btn_start = ctk.CTkButton(self.control_frame, text="Start Vision", fg_color="green", hover_color="darkgreen", command=self.start_vision)
        self.btn_start.pack(side="left", padx=10, pady=10, expand=True)

        self.btn_stop = ctk.CTkButton(self.control_frame, text="Stop Vision", fg_color="red", hover_color="darkred", state="disabled", command=self.stop_vision)
        self.btn_stop.pack(side="right", padx=10, pady=10, expand=True)

        # Robot Goals Frame
        self.goals_frame = ctk.CTkFrame(self.main_container)
        self.goals_frame.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        
        ctk.CTkLabel(self.goals_frame, text="Robot Goals & Directives", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, pady=5)
        
        self.goal_var = ctk.StringVar(value="Avoid All Objects")
        self.opt_goal = ctk.CTkOptionMenu(self.goals_frame, variable=self.goal_var, 
                                          values=["Avoid All Objects", "Follow Object", "Search for Object"], 
                                          command=self.set_active_goal)
        self.opt_goal.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.entry_target = ctk.CTkEntry(self.goals_frame, placeholder_text="Target Object/Class (e.g. person)")
        self.entry_target.insert(0, "person")
        self.entry_target.bind("<KeyRelease>", self.update_target_obj)
        self.entry_target.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        # Max FWD / REV sliders
        self.fwd_sld_label = ctk.CTkLabel(self.goals_frame, text="Max Forward Speed: 100%")
        self.fwd_sld_label.grid(row=2, column=0, padx=10, pady=(10,0))
        self.fwd_sld = ctk.CTkSlider(self.goals_frame, from_=0, to=100, command=self.update_fwd_limit)
        self.fwd_sld.set(100)
        self.fwd_sld.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        self.rev_sld_label = ctk.CTkLabel(self.goals_frame, text="Max Reverse Speed: 100%")
        self.rev_sld_label.grid(row=2, column=1, padx=10, pady=(10,0))
        self.rev_sld = ctk.CTkSlider(self.goals_frame, from_=0, to=100, command=self.update_rev_limit)
        self.rev_sld.set(100)
        self.rev_sld.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

        self.chk_looking = ctk.CTkCheckBox(self.goals_frame, text="Looking Mode", command=self.toggle_looking_mode)
        self.chk_looking.grid(row=4, column=0, padx=10, pady=(4, 8), sticky="w")

        self.lbl_looking_debug = ctk.CTkLabel(
            self.goals_frame,
            text="dbg req:OFF act:OFF seq:idle",
            font=ctk.CTkFont(size=11),
            text_color="gray70",
            anchor="e",
        )
        self.lbl_looking_debug.grid(row=4, column=1, padx=10, pady=(4, 8), sticky="e")
        
        self.goals_frame.columnconfigure(0, weight=1)
        self.goals_frame.columnconfigure(1, weight=1)
        self.update_looking_debug_line()

        # Telemetry Data Frame
        self.telemetry_frame = ctk.CTkFrame(self.main_container)
        self.telemetry_frame.grid(row=4, column=0, padx=20, pady=5, sticky="nsew")

        ctk.CTkLabel(self.telemetry_frame, text="Robot Brain Telemetry", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)

        self.lbl_directive = ctk.CTkLabel(self.telemetry_frame, text="CURRENT DIRECTIVE:\n[ STANDBY ]", font=ctk.CTkFont(size=14, weight="bold"), text_color="cyan")
        self.lbl_directive.pack(pady=10)

        self.lbl_threat_matrix = ctk.CTkLabel(self.telemetry_frame, text="THREAT MATRIX\nL: 0   |   C: 0   |   R: 0", font=ctk.CTkFont(size=13))
        self.lbl_threat_matrix.pack(pady=5)
        
        self.lbl_perf = ctk.CTkLabel(self.telemetry_frame, text="FPS: 0  |  Objects: 0", font=ctk.CTkFont(size=12))
        self.lbl_perf.pack(pady=5)

        self.lbl_entities = ctk.CTkLabel(self.telemetry_frame, text="TRACKED ENTITIES:\n- None", justify="left")
        self.lbl_entities.pack(pady=5, padx=10, anchor="w")

        # Visualization Frame limits
        self.viz_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.viz_frame.grid(row=5, column=0, padx=20, pady=5, sticky="nsew")

        # Joystick Frame (Left)
        self.joystick_frame = ctk.CTkFrame(self.viz_frame)
        self.joystick_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        ctk.CTkLabel(self.joystick_frame, text="Motor Vector", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)

        # Custom Tkinter Canvas for the Joystick Visual
        self.canvas_joy = tk.Canvas(self.joystick_frame, width=220, height=220, bg="#2b2b2b", highlightthickness=0)
        self.canvas_joy.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas_joy.bind("<Configure>", self.on_joy_resize)

        # Radar Frame (Right)
        self.radar_frame = ctk.CTkFrame(self.viz_frame)
        self.radar_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        ctk.CTkLabel(self.radar_frame, text="Predictive Path", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.canvas_radar = tk.Canvas(self.radar_frame, width=220, height=220, bg="#111111", highlightthickness=0)
        self.canvas_radar.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas_radar.bind("<Configure>", self.on_radar_resize)

        self.robot_base_image = None
        self.robot_img = None
        try:
            self.robot_base_image = Image.open("robot-topdown.png").convert("RGBA")
        except Exception:
            self.robot_base_image = None

        self.lbl_motor_data = ctk.CTkLabel(self.main_container, text="Power: 0% | Angle: 0%", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_motor_data.grid(row=6, column=0, padx=20, pady=(5, 10), sticky="ew")

        self.viz_frame.columnconfigure(0, weight=1)
        self.viz_frame.columnconfigure(1, weight=1)
        self.viz_frame.bind("<Configure>", lambda e: self.update_viz_layout(e.width))
        self.draw_joy_base()
        self.draw_radar_base()
        self.after(0, self.update_viz_layout)

    def on_window_resize(self, event=None):
        if event is not None and event.widget is not self:
            return

        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 1 or height <= 1:
            return

        scale = min(width / self.base_window_width, height / self.base_window_height)
        scale = max(0.50, min(1.35, scale))

        if abs(scale - self._ui_scale) >= 0.03:
            self._ui_scale = scale
            ctk.set_widget_scaling(scale)

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

        stacked = width < 420
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
            dot_radius = max(4, int(self.joy_radius * 0.10))
            dot_x = self.joy_cx + int(fwd_mag * self.joy_radius)
            dot_y = self.joy_cy - int(turn_mag * self.joy_radius)
            self.canvas_joy.coords(self.joy_dot, dot_x - dot_radius, dot_y - dot_radius, dot_x + dot_radius, dot_y + dot_radius)

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
        self.max_fwd_pct = max(0.0, float(value) / 100.0)
        self.fwd_sld_label.configure(text=f"Max Forward Speed: {int(value)}%")

    def update_rev_limit(self, value):
        self.max_rev_pct = max(0.0, float(value) / 100.0)
        self.rev_sld_label.configure(text=f"Max Reverse Speed: {int(value)}%")

    def toggle_looking_mode(self):
        self.looking_mode_requested = bool(self.chk_looking.get())
        self.update_looking_debug_line()

    def set_looking_mode(self, enabled):
        enabled = bool(enabled)
        if self.looking_mode == enabled:
            return
        self.looking_mode = enabled
        self.update_looking_debug_line()

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

    def set_uv_spot(self):
        self.withdraw()

        def on_selected(region):
            self.uv_spot = self._region_center(region)
            self.lbl_uv_spot.configure(text=f"UV Spot: ({self.uv_spot[0]}, {self.uv_spot[1]})")

        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    def set_tripod_spot(self):
        self.withdraw()

        def on_selected(region):
            self.tripod_spot = self._region_center(region)
            self.lbl_tripod_spot.configure(text=f"Tripod Spot: ({self.tripod_spot[0]}, {self.tripod_spot[1]})")

        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    def set_hold_position_spot(self):
        self.withdraw()

        def on_selected(region):
            self.hold_position_spot = self._region_center(region)
            self.lbl_hold_pos_spot.configure(text=f"Hold Position Spot: ({self.hold_position_spot[0]}, {self.hold_position_spot[1]})")

        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    def set_rear_most_spot(self):
        self.withdraw()

        def on_selected(region):
            self.rear_most_spot = self._region_center(region)
            self.lbl_rear_spot.configure(text=f"Rear-most Spot: ({self.rear_most_spot[0]}, {self.rear_most_spot[1]})")

        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    def set_zw_spot(self):
        self.withdraw()

        def on_selected(region):
            self.zw_spot = self._region_center(region)
            self.lbl_zw_spot.configure(text=f"ZW Spot: ({self.zw_spot[0]}, {self.zw_spot[1]})")

        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    def set_focus_window_spot(self):
        self.withdraw()

        def on_selected(region):
            self.focus_window_spot = self._region_center(region)
            self.lbl_focus_spot.configure(text=f"Focus Window: ({self.focus_window_spot[0]}, {self.focus_window_spot[1]})")

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

    def toggle_controller(self):
        self.translate_motion = bool(self.chk_controller.get())
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
            # use a segmentation model to get precise masks for convex hulls
            self.model = YOLO("yolov8n-seg.pt")
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
        cv2.namedWindow("Hex-Vision Output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hex-Vision Output", 640, 480)  # Reduced from 800x600
        
        sct = mss.mss()
        prev_time = time.time()
        
        # Smoothing variables for motor control
        smoothed_fwd = 0.0
        smoothed_turn = 0.0
        
        # Tracking variables
        last_target_cx = None
        last_follow_seen_time = 0.0
        follow_persist_sec = 1.0
        persisted_follow_base_fwd = 0.0
        persisted_follow_turn = 0.0
        estimated_world_dx = 0.0
        
        # Load ctypes for global Escape monitor
        try:
            import ctypes
            user32 = ctypes.windll.user32
        except ImportError:
            user32 = None
        esc_was_down = False

        while self.is_running:
            # Global Escape disables mouse translation without exiting the app.
            esc_down = bool(user32 and (user32.GetAsyncKeyState(0x1B) & 0x8000))
            if esc_down and not esc_was_down:
                if self.translate_motion:
                    print("Escape detected: disabling mouse translation.")
                    self.translate_motion = False
                    self.mouse_hold_arm_time = None
                    self.set_mouse_hold(False)
                    self.after(0, self.chk_controller.deselect)
            esc_was_down = esc_down
                
            # Measure FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time
            
            detected_objects = []
            
            # Capture RGB screen
            rgb_screenshot = sct.grab(self.rgb_monitor)
            frame = np.array(rgb_screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Capture Depth screen if configured
            depth_frame = None
            if self.depth_monitor:
                depth_screenshot = sct.grab(self.depth_monitor)
                depth_frame_bgra = np.array(depth_screenshot)
                # Convert Depth to grayscale, but we pull the RED channel purely 
                # (Index 2 in BGRA) since sometimes depth cameras output red to mean "pure white / close".
                depth_frame = depth_frame_bgra[:, :, 2]
                
                # Resize depth exactly to match RGB width and height
                depth_frame = cv2.resize(depth_frame, (frame.shape[1], frame.shape[0]))

                # --- GENERIC DETECTOR: Catch ANY Close Obstruction on Depth Map ---
                # Threshold to detect any pixels > 130 intensity (close obstacles)
                _, thresh = cv2.threshold(depth_frame, 130, 255, cv2.THRESH_BINARY)
                obstruction_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for c in obstruction_contours:
                    area = cv2.contourArea(c)
                    if area > 600:  # Ignore small noise spikes
                        ob_hull = cv2.convexHull(c)
                        # Check actual average density of this depth obstacle
                        mask_single = np.zeros_like(depth_frame)
                        cv2.drawContours(mask_single, [ob_hull], -1, 255, thickness=cv2.FILLED)
                        obs_mean = cv2.mean(depth_frame, mask=mask_single)[0]
                        
                        obs_color = (0, 255, 255) # Default yellow-orange
                        obs_label = "Obstruction"
                        
                        if obs_mean > 220:
                            obs_label = "! CRITICAL COLLISION !"
                            obs_color = (0, 0, 255) # Bright Red
                            cv2.drawContours(frame, [ob_hull], -1, obs_color, 4) # Thicker boundary for critical
                        elif obs_mean > 175:
                            obs_label = "SEVERE OBSTRUCTION"
                            obs_color = (0, 69, 255) # Red-Orange
                            cv2.drawContours(frame, [ob_hull], -1, obs_color, 3)
                        elif obs_mean > 130:
                            obs_label = "Approaching Surface"
                            obs_color = (0, 140, 255) # Orange
                            cv2.drawContours(frame, [ob_hull], -1, obs_color, 2)

                        # Render general obstruction text tag
                        M = cv2.moments(ob_hull)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            
                            obs_text = f"{obs_label} (Z: {int(obs_mean)})"
                            
                            # Render via Pillow for TTF Consolas
                            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_frame)
                            try:
                                font = ImageFont.truetype("orbitron.ttf", 20)
                            except IOError:
                                font = ImageFont.load_default()
                                
                            draw.text((cX - 40, cY - 25), obs_text, font=font, fill=(obs_color[2], obs_color[1], obs_color[0]))
                            frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
                            
                            detected_objects.append(f"Depth Mass [Z: {int(obs_mean)}]")

            target_cx = None
            target_depth = None
            
            # Perform inference on RGB
            results = self.model(frame, verbose=False)
            
            # Draw convex hulls
            for r in results:
                if r.masks is not None:
                    # Get masks and extract contours
                    masks = r.masks.data.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()
                    names = self.model.names
                    
                    for idx, mask in enumerate(masks):
                        # Resize mask to original frame shape
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        
                        # Find contours
                        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            # Largest contour
                            largest_contour = max(contours, key=cv2.contourArea)
                            if cv2.contourArea(largest_contour) > 100: # Filter tiny specks
                                hull = cv2.convexHull(largest_contour)
                                
                                # Process distance Data if depth is available
                                distance_text = ""
                                text_color = (0, 255, 255)
                                mean_depth = 0
                                if depth_frame is not None:
                                    # Create an empty mask in shape of frame to isolate the actual object in depth frame
                                    obj_mask = np.zeros_like(depth_frame)
                                    cv2.drawContours(obj_mask, [hull], -1, 255, thickness=cv2.FILLED)
                                    
                                    # Collect pixel intensities inside the masked hull
                                    mean_depth = cv2.mean(depth_frame, mask=obj_mask)[0]
                                    
                                    # More granular mapping for recognized YOLO objects
                                    if mean_depth > 220:
                                        distance_text = "- CRITICAL COLLISION"
                                        text_color = (0, 0, 255) # Red text
                                        cv2.drawContours(frame, [hull], -1, (0, 0, 255), 4) # Flash thick red hull
                                    elif mean_depth > 175:
                                        distance_text = "- SEVERE OBSTACLE"
                                        text_color = (0, 69, 255) # Red-Orange
                                        cv2.drawContours(frame, [hull], -1, (0, 69, 255), 3)
                                    elif mean_depth > 130:
                                        distance_text = "- Approaching"
                                        text_color = (0, 140, 255) # Orange
                                        cv2.drawContours(frame, [hull], -1, (0, 140, 255), 2)
                                    elif mean_depth > 80:
                                        distance_text = "- Near"
                                        text_color = (0, 255, 255) # Yellow
                                        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2) # Green Hull
                                    else:
                                        distance_text = "- Far/Safe"
                                        text_color = (0, 255, 0) # Green
                                        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2) # Green Hull
                                else:
                                    # Fallback contours if no depth is configured
                                    cv2.drawContours(frame, [hull], -1, (255, 255, 255), 2) # White bounding box
                                    text_color = (255, 255, 255)
                                    distance_text = ""

                                # Label
                                class_name = names[int(classes[idx])]
                                M = cv2.moments(hull)
                                if M["m00"] != 0:
                                    cX = int(M["m10"] / M["m00"])
                                    cY = int(M["m01"] / M["m00"])
                                    
                                    if self.active_goal == "Follow Object" and class_name.lower() == self.target_object:
                                        target_cx = cX
                                        target_depth = mean_depth
                                        cv2.circle(frame, (cX, cY), 5, (255, 0, 255), -1) # Highlight target smaller circle
                                        
                                    label_text = f"{class_name} {distance_text}"
                                    
                                    # Use Pil / TTF for nice tech font
                                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                    draw = ImageDraw.Draw(pil_frame)
                                    try:
                                        font = ImageFont.truetype("orbitron.ttf", 20)
                                    except IOError:
                                        font = ImageFont.load_default()
                                        
                                    draw.text((cX - 20, cY - 20), label_text, font=font, fill=(text_color[2], text_color[1], text_color[0]))
                                    frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
                                    
                                    detected_objects.append(label_text)

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
                    if target_cx is not None:
                        # We see the target!
                        action = f"FOLLOWING: {self.target_object.upper()}"
                        action_color = (0, 255, 0)
                        
                        # Determine turn based on horizontal position
                        # Center is width_f / 2
                        center_x = width_f / 2
                        half_width = width_f / 2
                        raw_offset = target_cx - center_x

                        # Compensate apparent target drift introduced by our own turn command.
                        camera_offset_comp_px = smoothed_turn * half_width * self.turn_comp_gain
                        compensated_offset = raw_offset + camera_offset_comp_px
                        turn_mag = max(-1.0, min(1.0, compensated_offset / half_width))

                        if last_target_cx is not None:
                            raw_dx = target_cx - last_target_cx
                            camera_dx = -smoothed_turn * half_width * self.turn_comp_gain
                            world_dx = raw_dx - camera_dx
                            estimated_world_dx = (0.2 * world_dx) + (0.8 * estimated_world_dx)
                        last_target_cx = target_cx
                        
                        want_looking_mode = self.looking_mode_requested or (target_depth is not None and target_depth >= self.super_close_threshold)

                        if want_looking_mode:
                            # Looking mode: run hold->UV->rear->ZW sequence, then look only with turn.
                            base_fwd_mag = 0.0
                            seq_ready, seq_text, seq_color = self.run_follow_enter_sequence()
                            action = f"{seq_text}: {self.target_object.upper()}"
                            action_color = seq_color
                            if not seq_ready:
                                turn_mag = 0.0
                        else:
                            # Not super close: exit sequence (hold off -> tripod), then normal follow.
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

                        last_follow_seen_time = curr_time
                        persisted_follow_base_fwd = base_fwd_mag
                        persisted_follow_turn = turn_mag
                    else:
                        time_since_seen = curr_time - last_follow_seen_time if last_follow_seen_time > 0 else 999.0
                        if time_since_seen <= follow_persist_sec:
                            action = f"PERSISTING TO LAST: {self.target_object.upper()}"
                            action_color = (0, 255, 255)
                            fwd_mag = persisted_follow_base_fwd
                            turn_mag = max(-1.0, min(1.0, persisted_follow_turn + (estimated_world_dx / max(1.0, width_f * 0.5))))
                        else:
                            action = f"WAITING FOR: {self.target_object.upper()}"
                            action_color = (150, 150, 150)
                            fwd_mag = 0.0
                            turn_mag = 0.0
                            last_target_cx = None
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
            
            # Apply Exponential Moving Average (EMA) smoothing to motor commands
            # This drastically stabilizes the highly flawed/flickering input feed 
            alpha = 0.08  # 8% new reactivity, 92% history
            smoothed_fwd = (alpha * fwd_mag) + ((1.0 - alpha) * smoothed_fwd)
            smoothed_turn = (alpha * turn_mag) + ((1.0 - alpha) * smoothed_turn)
            
            # push to UI
            self.after(0, self.update_telemetry, action, b_color_tk, threat_text, perf_text, entities_text, smoothed_fwd, smoothed_turn)
            
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

            # Show window
            cv2.imshow("Hex-Vision Output", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_vision()
                break

        # Safely destroy windows when loop terminates
        self.mouse_hold_arm_time = None
        self.set_mouse_hold(False)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HexVisionApp()
    app.mainloop()
