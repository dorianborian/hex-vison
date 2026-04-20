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
        self.resizable(False, False)

        # state
        self.is_running = False
        self.capture_thread = None
        
        # Robot Goal states
        self.active_goal = "Avoid All Objects"
        self.target_object = "person"
        self.max_fwd_pct = 1.0
        self.max_rev_pct = 1.0
        
        # default screen regions
        self.rgb_monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
        self.depth_monitor = None # User must set this if they want depth scanning
        self.controller_monitor = None # User must set this to output motion vectors
        self.translate_motion = False
        
        # Load YOLOv8 segmentation model
        self.model = None

        self.setup_ui()

    def setup_ui(self):
        self.label = ctk.CTkLabel(self, text="Hex-Vision Object Tracker", font=ctk.CTkFont(size=20, weight="bold"))
        self.label.pack(pady=20)

        # Region Frame
        self.region_frame = ctk.CTkFrame(self)
        self.region_frame.pack(pady=10, padx=20, fill="x")

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

        # Control Frame
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(pady=10, padx=20, fill="x")

        self.btn_start = ctk.CTkButton(self.control_frame, text="Start Vision", fg_color="green", hover_color="darkgreen", command=self.start_vision)
        self.btn_start.pack(side="left", padx=10, pady=10, expand=True)

        self.btn_stop = ctk.CTkButton(self.control_frame, text="Stop Vision", fg_color="red", hover_color="darkred", state="disabled", command=self.stop_vision)
        self.btn_stop.pack(side="right", padx=10, pady=10, expand=True)

        # Robot Goals Frame
        self.goals_frame = ctk.CTkFrame(self)
        self.goals_frame.pack(pady=5, padx=20, fill="x")
        
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
        
        self.goals_frame.columnconfigure(0, weight=1)
        self.goals_frame.columnconfigure(1, weight=1)

        # Telemetry Data Frame
        self.telemetry_frame = ctk.CTkFrame(self)
        self.telemetry_frame.pack(pady=5, padx=20, fill="both", expand=True)

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
        self.viz_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.viz_frame.pack(pady=5, padx=20, fill="both", expand=True)

        # Joystick Frame (Left)
        self.joystick_frame = ctk.CTkFrame(self.viz_frame)
        self.joystick_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        ctk.CTkLabel(self.joystick_frame, text="Motor Vector", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)

        # Custom Tkinter Canvas for the Joystick Visual
        self.canvas_joy = tk.Canvas(self.joystick_frame, width=160, height=160, bg="#2b2b2b", highlightthickness=0)
        self.canvas_joy.pack(pady=10)

        # Draw coordinate grid and circle
        self.canvas_joy.create_oval(10, 10, 150, 150, outline="#555555", width=2)   # Boundary Circle
        self.canvas_joy.create_line(80, 10, 80, 150, fill="#444444", dash=(4, 4))   # Y Axis
        self.canvas_joy.create_line(10, 80, 150, 80, fill="#444444", dash=(4, 4))   # X Axis
        
        # The moving control point
        self.joy_dot = self.canvas_joy.create_oval(75, 75, 85, 85, fill="cyan")

        # Radar Frame (Right)
        self.radar_frame = ctk.CTkFrame(self.viz_frame)
        self.radar_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        ctk.CTkLabel(self.radar_frame, text="Predictive Path", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.canvas_radar = tk.Canvas(self.radar_frame, width=160, height=160, bg="#111111", highlightthickness=0)
        self.canvas_radar.pack(pady=10)
        
        # Base Radar UI
        self.canvas_radar.create_line(80, 0, 80, 160, fill="#333333", dash=(2, 4))
        self.canvas_radar.create_line(0, 130, 160, 130, fill="#333333", dash=(4, 4))
        self.canvas_radar.create_arc(40, 90, 120, 170, outline="#333333", start=0, extent=180)
        self.canvas_radar.create_arc(10, 60, 150, 200, outline="#333333", start=0, extent=180)
        
        try:
            pil_img = Image.open("robot-topdown.png").resize((40, 40))
            self.robot_img = ImageTk.PhotoImage(pil_img)
            self.canvas_radar.create_image(80, 130, image=self.robot_img, tag="robot")
        except Exception:
            self.canvas_radar.create_rectangle(65, 115, 95, 145, fill="gray", tag="robot")

        self.lbl_motor_data = ctk.CTkLabel(self, text="Power: 0% | Angle: 0%", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_motor_data.pack(pady=10)

        # -----------------------------
        # Section 5: Robotic Goals & Limits
        # -----------------------------
        self.goals_frame = ctk.CTkFrame(self)
        self.goals_frame.pack(pady=5, padx=10, fill="x")

        goal_label = ctk.CTkLabel(self.goals_frame, text="Current Goal State:", font=("Consolas", 14, "bold"))
        goal_label.pack(pady=(5, 0))

        self.goal_var = ctk.StringVar(value="Avoid All Objects")
        goal_menu = ctk.CTkOptionMenu(self.goals_frame, variable=self.goal_var,
                                      values=["Avoid All Objects", "Follow Object", "Search Space"],
                                      command=self.set_active_goal)
        goal_menu.pack(pady=5)
        self.active_goal = self.goal_var.get()

        obj_label = ctk.CTkLabel(self.goals_frame, text="Target Object:")
        obj_label.pack()
        self.entry_target = ctk.CTkEntry(self.goals_frame, placeholder_text="e.g. cup, person")
        self.entry_target.pack(pady=5)
        self.entry_target.bind("<Return>", self.update_target_obj)
        self.entry_target.bind("<FocusOut>", self.update_target_obj)
        self.target_object = ""

        # Throttle sliders
        self.fwd_sld_label = ctk.CTkLabel(self.goals_frame, text="Max Forward Speed: 100%")
        self.fwd_sld_label.pack()
        self.fwd_sld = ctk.CTkSlider(self.goals_frame, from_=0, to=100, command=self.update_fwd_limit)
        self.fwd_sld.set(100)
        self.fwd_sld.pack(pady=5)
        self.max_fwd_pct = 1.0

        self.rev_sld_label = ctk.CTkLabel(self.goals_frame, text="Max Reverse Speed: 100%")
        self.rev_sld_label.pack()
        self.rev_sld = ctk.CTkSlider(self.goals_frame, from_=0, to=100, command=self.update_rev_limit)
        self.rev_sld.set(100)
        self.rev_sld.pack(pady=(0, 10))
        self.max_rev_pct = 1.0

    def set_active_goal(self, value):
        self.active_goal = value
        
    def update_target_obj(self, event=None):
        self.target_object = self.entry_target.get().lower().strip()

    def update_fwd_limit(self, value):
        self.max_fwd_pct = max(0.0, float(value) / 100.0)
        self.fwd_sld_label.configure(text=f"Max Forward Speed: {int(value)}%")

    def update_rev_limit(self, value):
        self.max_rev_pct = max(0.0, float(value) / 100.0)
        self.rev_sld_label.configure(text=f"Max Reverse Speed: {int(value)}%")

    def update_telemetry(self, directive_text, directive_color, threat_text, perf_text, entities_text, fwd_mag, turn_mag):
        self.lbl_directive.configure(text=f"CURRENT DIRECTIVE:\n[ {directive_text} ]", text_color=directive_color)
        self.lbl_threat_matrix.configure(text=threat_text)
        self.lbl_perf.configure(text=perf_text)
        self.lbl_entities.configure(text=entities_text)

        # Update Joystick position
        # fwd_mag: -1 (Left/Forward) to 1 (Right/Reverse)
        # turn_mag: 1 (Up/TurnR) to -1 (Down/TurnL) -- tk canvas Up is negative Y
        cx, cy = 80, 80
        radius = 70
        dot_x = cx + int(fwd_mag * radius)
        dot_y = cy - int(turn_mag * radius)
        
        self.canvas_joy.coords(self.joy_dot, dot_x - 5, dot_y - 5, dot_x + 5, dot_y + 5)
        
        # Format textual motor output
        speed_pct = int(-fwd_mag * 100) # Negative fwd_mag is Forward (+ Speed)
        dir_txt = "Fwd" if speed_pct > 0 else "Rev" if speed_pct < 0 else "Stop"
        
        turn_pct = int(turn_mag * 100)  # Positive turn_mag is Right
        turn_txt = "Right" if turn_pct > 0 else "Left" if turn_pct < 0 else "Straight"
        
        self.lbl_motor_data.configure(text=f"Power: {abs(speed_pct)}% {dir_txt}  |  Angle: {abs(turn_pct)}% {turn_txt}")
        
        # --- Update Predictive Radar Path ---
        self.canvas_radar.delete("path")
        rx, ry = 80, 130 # Robot anchor point
        
        # scale vectors visually
        speed_px = speed_pct * 1.2 # Maps 100% to ~120px out
        curve_px = turn_pct * 0.8  # Maps 100% to ~80px wide
        
        if abs(speed_px) > 2:
            # 3-Point Bezier Spline
            pt1 = (rx, ry)
            pt2 = (rx + (curve_px * 0.5), ry - (speed_px * 0.5))
            pt3 = (rx + curve_px, ry - speed_px)
            
            path_color = "green" if speed_pct > 0 else "red"
            self.canvas_radar.create_line(pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1], 
                                          smooth=True, fill=path_color, width=3, arrow=tk.LAST, tag="path")
        
        self.canvas_radar.tag_raise("robot")

    def set_active_goal(self, value):
        self.active_goal = value
        
    def update_target_obj(self, event=None):
        self.target_object = self.entry_target.get().lower().strip()

    def update_fwd_limit(self, value):
        self.max_fwd_pct = max(0.0, float(value) / 100.0)
        self.fwd_sld_label.configure(text=f"Max Forward Speed: {int(value)}%")

    def update_rev_limit(self, value):
        self.max_rev_pct = max(0.0, float(value) / 100.0)
        self.rev_sld_label.configure(text=f"Max Reverse Speed: {int(value)}%")

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
            
        selector = RegionSelector(self, on_selected)
        self.wait_window(selector)
        self.deiconify()

    def toggle_controller(self):
        self.translate_motion = bool(self.chk_controller.get())
        import ctypes
        if self.translate_motion and self.controller_monitor:
            # issue Down when starting
            ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0) # MOUSEEVENTF_LEFTDOWN
        elif not self.translate_motion:
            # issue Up when stopping
            ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0) # MOUSEEVENTF_LEFTUP

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
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_set_rgb.configure(state="disabled")
        self.btn_set_depth.configure(state="disabled")

        self.capture_thread = threading.Thread(target=self.vision_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def stop_vision(self):
        self.is_running = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_set_rgb.configure(state="normal")
        self.btn_set_depth.configure(state="normal")

    def vision_loop(self):
        cv2.namedWindow("Hex-Vision Output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hex-Vision Output", 800, 600)  # Default to a smaller, reasonable window size
        
        sct = mss.mss()
        prev_time = time.time()
        
        # Smoothing variables for motor control
        smoothed_fwd = 0.0
        smoothed_turn = 0.0
        
        # Tracking variables
        last_target_cx = None
        smoothed_target_dx = 0.0
        
        # Load ctypes for global Escape monitor
        try:
            import ctypes
            user32 = ctypes.windll.user32
        except ImportError:
            user32 = None

        while self.is_running:
            # Check for global Escape key (virtual key code 0x1B)
            if user32 and user32.GetAsyncKeyState(0x1B) & 0x8000:
                print("Global Escape detected, shutting down...")
                self.stop_vision()
                # Use after to guarantee safe UI destruction in main thread
                self.after(0, self.destroy)
                break
                
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
                if self.active_goal == "Follow Object" and self.target_object:
                    if target_cx is not None:
                        # Update tracking delta (direction of movement)
                        if last_target_cx is not None:
                            dx = target_cx - last_target_cx
                            alpha_tracker = 0.15 # 15% new, 85% old for movement vector
                            smoothed_target_dx = (alpha_tracker * dx) + ((1.0 - alpha_tracker) * smoothed_target_dx)
                        
                        last_target_cx = target_cx

                        # We see the target!
                        action = f"FOLLOWING: {self.target_object.upper()}"
                        action_color = (0, 255, 0)
                        
                        # Determine turn based on horizontal position
                        # Center is width_f / 2
                        center_x = width_f / 2
                        offset = target_cx - center_x
                        turn_mag = max(-1.0, min(1.0, offset / (width_f / 2)))
                        
                        # Determine fwd based on depth (we want it to be ~130 distance)
                        if target_depth is not None:
                            if target_depth > 180:
                                # Too close, reverse!
                                fwd_mag = 1.0 # Reverse
                                action = f"BACKING UP FROM: {self.target_object.upper()}"
                                action_color = (0, 0, 255)
                            elif target_depth > 130:
                                # Good distance, stop
                                fwd_mag = 0.0
                                action = f"HOLDING AT: {self.target_object.upper()}"
                                action_color = (0, 255, 255)
                            else:
                                # Move forward to follow
                                fwd_mag = -1.0 # Forward
                        else:
                            fwd_mag = 0.0
                    else:
                        action = f"SEARCHING: {self.target_object.upper()}"
                        action_color = (0, 165, 255)
                        fwd_mag = 0.0
                        
                        # Look in the last known direction the target was moving
                        # Default is right (0.5), but we update based on smoothed dx
                        if smoothed_target_dx > 2.0:
                            turn_mag = 0.6  # Spin Right
                        elif smoothed_target_dx < -2.0:
                            turn_mag = -0.6 # Spin Left
                        else:
                            turn_mag = 0.5  # Default slow spin to find it if we don't have enough data
                            
                        # If target is lost for a while, we clear out the tracking history softly via multiplying
                        # or we just keep the last known velocity to keep spinning in that circle.
                        last_target_cx = None
                        
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
                action = "WAITING FOR DEPTH FEED"
                action_color = (150, 150, 150)
                fwd_mag = 0.0
                turn_mag = 0.0

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

            # Show window
            cv2.imshow("Hex-Vision Output", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_vision()
                break

        # Safely destroy windows when loop terminates
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HexVisionApp()
    app.mainloop()
