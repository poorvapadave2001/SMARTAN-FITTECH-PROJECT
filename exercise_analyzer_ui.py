import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import json
import time

class ExerciseAnalyzerUI:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.root = tk.Tk()
        self.root.title("Exercise Form Analyzer")
        
        # Set fixed window size with scrollable area
        self.root.geometry("850x700")
        self.root.minsize(850, 700)
        
        # Create main container with scrollbar
        self.main_canvas = tk.Canvas(self.root)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.main_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.main_canvas.bind('<Configure>', lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")))
        
        self.main_frame = ttk.Frame(self.main_canvas)
        self.main_canvas.create_window((0,0), window=self.main_frame, anchor="nw")
        
        # Video display (fixed size)
        self.video_frame = ttk.LabelFrame(self.main_frame, text="Video Feed", width=800, height=400)
        self.video_frame.pack_propagate(False)  # Prevent frame from resizing to contents
        self.video_frame.pack(pady=5)
        
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control panel (fixed height)
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls", height=150)
        self.control_frame.pack_propagate(False)
        self.control_frame.pack(fill=tk.X, pady=5)
        
        # Exercise selection
        self.exercise_frame = ttk.Frame(self.control_frame)
        self.exercise_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        ttk.Label(self.exercise_frame, text="Select Exercise:").pack(anchor=tk.W)
        self.exercise_var = tk.StringVar()
        self.exercise_dropdown = ttk.Combobox(
            self.exercise_frame,
            textvariable=self.exercise_var,
            values=list(self.analyzer.feedback_analyzer.exercise_rules.keys()),
            state="readonly",
            width=25
        )
        self.exercise_dropdown.pack(anchor=tk.W)
        self.exercise_dropdown.set("Select Exercise")
        
        # Source selection
        self.source_frame = ttk.Frame(self.control_frame)
        self.source_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        ttk.Label(self.source_frame, text="Input Source:").pack(anchor=tk.W)
        self.source_var = tk.StringVar(value="webcam")
        ttk.Radiobutton(
            self.source_frame,
            text="Webcam",
            variable=self.source_var,
            value="webcam"
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            self.source_frame,
            text="Video File",
            variable=self.source_var,
            value="video"
        ).pack(anchor=tk.W)
        
        # Action buttons
        self.button_frame = ttk.Frame(self.control_frame)
        self.button_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.start_button = ttk.Button(
            self.button_frame,
            text="Start Analysis",
            command=self.start_analysis,
            width=15
        )
        self.start_button.pack(pady=2)
        
        self.stop_button = ttk.Button(
            self.button_frame,
            text="Stop",
            command=self.stop_analysis,
            state=tk.DISABLED,
            width=15
        )
        self.stop_button.pack(pady=2)
        
        self.export_button = ttk.Button(
            self.button_frame,
            text="Export Data",
            command=self.export_data,
            state=tk.DISABLED,
            width=15
        )
        self.export_button.pack(pady=2)
        
        # Progress bar
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, expand=True)
        
        # Feedback display (scrollable)
        self.feedback_frame = ttk.LabelFrame(self.main_frame, text="Exercise Feedback", height=150)
        self.feedback_frame.pack_propagate(False)
        self.feedback_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.feedback_text = tk.Text(self.feedback_frame, wrap=tk.WORD)
        self.feedback_scroll = ttk.Scrollbar(self.feedback_frame, command=self.feedback_text.yview)
        self.feedback_text.configure(yscrollcommand=self.feedback_scroll.set)
        
        self.feedback_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.feedback_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text colors
        self.feedback_text.tag_config('good', foreground='green')
        self.feedback_text.tag_config('medium', foreground='orange')
        self.feedback_text.tag_config('bad', foreground='red')
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN
        )
        self.status_bar.pack(fill=tk.X, pady=(5,0))
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.video_path = ""
        self.analysis_data = []

    # [Rest of the methods remain unchanged...]
        
        # Configure grid weights for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=1)
        
    def start_analysis(self):
        exercise = self.exercise_var.get()
        if not exercise or exercise == "Select Exercise":
            messagebox.showerror("Error", "Please select an exercise")
            return
            
        self.analyzer.current_exercise = exercise
        self.analyzer.is_running = True
        self.is_running = True
        self.analysis_data = []  # Reset analysis data
        
        source = self.source_var.get()
        if source == "webcam":
            self.status_var.set("Starting webcam...")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.export_button.config(state=tk.DISABLED)  # Disable until we have data
            self.process_webcam()
        else:
            self.video_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
            )
            if self.video_path:
                self.status_var.set(f"Processing video: {self.video_path}")
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.export_button.config(state=tk.DISABLED)
                
                # Clear previous data
                self.analysis_data = []
                self.feedback_text.delete(1.0, tk.END)
                self.progress['value'] = 0
                
                # Start video processing thread
                threading.Thread(
                    target=self.analyzer.process_video,
                    args=(self.video_path, self.update_ui),
                    daemon=True
                ).start()
                
    def update_ui(self, event_type, data):
    #"""Handle updates from video processing thread"""
        if event_type == "error":
            messagebox.showerror("Error", data)
            self.stop_analysis()
        elif event_type == "status":
            self.status_var.set(data)
        elif event_type == "progress":
            self.progress['value'] = data
            self.root.update()
        elif event_type == "frame":
            frame, frame_data = data
            self.analysis_data.append(frame_data)
        
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        
            if frame_data['persons']:
                self.update_feedback_display(
                    frame_data['frame_number'],
                    frame_data['persons'][0]['feedback']
                )   
        elif event_type == "complete":
            frames_processed = data["frames_processed"]
            output_files = data["output_files"]
        
            self.status_var.set(f"Analysis complete! Processed {frames_processed} frames")
            self.export_button.config(state=tk.NORMAL)  # Enable export
            self.stop_button.config(state=tk.DISABLED)
            self.start_button.config(state=tk.NORMAL)
        
            messagebox.showinfo(
                "Analysis Complete",
                f"Video analysis completed!\n\n"
                f"Data automatically saved to:\n"
                f"JSON: {output_files.get('json', 'Not saved')}\n"
                f"CSV: {output_files.get('csv', 'Not saved')}"
            )
            
    def update_feedback_display(self, frame_num, feedback_data):
        """Update the feedback text widget with current analysis"""
        self.feedback_text.delete(1.0, tk.END)
        
        self.feedback_text.insert(tk.END, f"Frame: {frame_num}\n")
        self.feedback_text.insert(tk.END, f"Overall Score: {feedback_data['overall_score']:.2f}\n\n")
        
        for feedback in feedback_data['feedback']:
            if feedback_data['overall_score'] >= 0.8:
                self.feedback_text.insert(tk.END, f"✓ {feedback}\n", 'good')
            elif feedback_data['overall_score'] >= 0.6:
                self.feedback_text.insert(tk.END, f"⚠ {feedback}\n", 'medium')
            else:
                self.feedback_text.insert(tk.END, f"✗ {feedback}\n", 'bad')
        
        # Configure text colors
        self.feedback_text.tag_config('good', foreground='green')
        self.feedback_text.tag_config('medium', foreground='orange')
        self.feedback_text.tag_config('bad', foreground='red')
    
    def export_data(self):
        """Export analysis data to file"""
        if not self.analysis_data:
            messagebox.showinfo("Info", "No analysis data to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("CSV Files", "*.csv")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump({
                            'source': 'webcam' if self.source_var.get() == 'webcam' else self.video_path,
                            'exercise': self.analyzer.current_exercise,
                            'analysis_data': self.analysis_data
                        }, f, indent=2)
                else:
                    import pandas as pd
                    # Flatten data for CSV
                    csv_data = []
                    for frame in self.analysis_data:
                        for person in frame['persons']:
                            csv_data.append({
                                'frame_number': frame['frame_number'],
                                'timestamp': frame['timestamp'],
                                'person_id': person['person_id'],
                                'overall_score': person['feedback']['overall_score'],
                                'feedback': "; ".join(person['feedback']['feedback'])
                            })
                    pd.DataFrame(csv_data).to_csv(file_path, index=False)
                
                messagebox.showinfo("Success", f"Data exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
    def stop_analysis(self):
    #"""Stop analysis and clean up resources"""
        self.is_running = False
        self.analyzer.is_running = False
        if self.cap:
            self.cap.release()
    
        # Enable Export button if we have analysis data
        if self.analysis_data:
            self.export_button.config(state=tk.NORMAL)
    
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Ready")

    
    def process_webcam(self):
    #"""Process webcam feed with automatic data collection"""
        if not self.is_running:
            return
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        def update_frame():
            ret, frame = self.cap.read()
            if ret and self.is_running:
                # Process frame
                results = self.analyzer.process_frame(frame)
                
                frame_data = {
                    'frame_number': len(self.analysis_data) + 1,
                    'timestamp': time.time(),
                    'persons': []
                }
                
                if results.pose_landmarks:
                    # For webcam demo, we'll process each detected pose
                    person_id = 0  # Single person for UI demo
                    
                    # Smooth Keypoints
                    smoothed_landmarks = self.analyzer.smooth_keypoints(results.pose_landmarks, person_id)
                    
                    # Frame-Wise Feedback Logic
                    feedback_data = self.analyzer.feedback_analyzer.analyze_exercise_form(
                        smoothed_landmarks, 
                        self.analyzer.current_exercise
                    )
                    
                    frame_person_data = {
                        'person_id': person_id,
                        'landmarks': [(lm.x, lm.y, lm.z) for lm in smoothed_landmarks],
                        'feedback': feedback_data
                    }
                    
                    self.analysis_data.append(frame_data)
                    frame_data['persons'].append(frame_person_data)
                    
                    # Show Feedback
                    frame = self.analyzer.feedback_analyzer.draw_feedback(
                        frame, 
                        {person_id: feedback_data}, 
                        0,
                        self.analyzer.current_exercise,
                        self.analyzer.person_colors
                    )
                    
                    # Draw pose landmarks
                    color = self.analyzer.person_colors[person_id % len(self.analyzer.person_colors)]
                    self.analyzer.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.analyzer.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.analyzer.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        connection_drawing_spec=self.analyzer.mp_drawing.DrawingSpec(color=color, thickness=2)
                    )
                
                # Update UI
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(display_frame)
                imgtk = ImageTk.PhotoImage(image=img)
            
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            
                if frame_data['persons']:
                    self.update_feedback_display(
                        frame_data['frame_number'],
                        frame_data['persons'][0]['feedback']
                    )
                
                # Schedule next update
                self.video_label.after(10, update_frame)
            else:
                self.stop_analysis()
    
        update_frame()
                
    def process_video(self):
        if not self.is_running:
            return
            
        self.cap = cv2.VideoCapture(self.video_path)
        
        def update_frame():
            ret, frame = self.cap.read()
            if ret and self.is_running:
                # Process frame
                results = self.analyzer.process_frame(frame)
                
                if results.pose_landmarks:
                    # For video demo, we'll process each detected pose
                    person_id = 0  # Single person for UI demo
                    
                    # Smooth Keypoints
                    smoothed_landmarks = self.analyzer.smooth_keypoints(results.pose_landmarks, person_id)
                    
                    # Frame-Wise Feedback Logic
                    feedback_data = self.analyzer.feedback_analyzer.analyze_exercise_form(
                        smoothed_landmarks, 
                        self.analyzer.current_exercise
                    )
                    
                    # Show Feedback
                    frame = self.analyzer.feedback_analyzer.draw_feedback(
                        frame, 
                        {person_id: feedback_data}, 
                        0,
                        self.analyzer.current_exercise,
                        self.analyzer.person_colors
                    )
                    
                    # Draw pose landmarks
                    color = self.analyzer.person_colors[person_id % len(self.analyzer.person_colors)]
                    self.analyzer.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.analyzer.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.analyzer.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        connection_drawing_spec=self.analyzer.mp_drawing.DrawingSpec(color=color, thickness=2)
                    )
                
                # Convert to PIL Image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update display
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
                
                # Schedule next update
                self.video_label.after(10, update_frame)
            else:
                self.stop_analysis()
        
        update_frame()
    
    def run(self):
        self.root.mainloop()