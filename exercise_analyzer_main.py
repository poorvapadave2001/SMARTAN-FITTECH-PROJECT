import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque
import json
from scipy.spatial.distance import cdist
from feedback_analysis import FeedbackAnalyzer

class MultiPersonExerciseFormAnalyzer:
    def __init__(self):
        # Step 1: Environment Setup with improved configuration
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.7,  # Increased for better detection
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Multi-person tracking setup
        self.person_tracker = PersonTracker()
        self.max_persons = 10  # Maximum number of people to track
        
        # Step 6: Smoothing setup (per person)
        self.smoothing_window = 5
        self.keypoint_history = {}  # Will store per person: {person_id: {keypoint_idx: deque}}
        
        # Step 9: Time-series data storage (per person)
        self.pose_data = {}  # Will store per person: {person_id: [frame_data]}
        self.feedback_history = {}
        
        # Exercise selection
        self.current_exercise = None
        self.feedback_analyzer = FeedbackAnalyzer()
        
        # Frame dimensions for MediaPipe optimization
        self.frame_width = None
        self.frame_height = None
        
        # Person colors for visualization (BGR format)
        self.person_colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        # Convert to numpy arrays
        p1 = np.array([point1.x, point1.y])
        p2 = np.array([point2.x, point2.y])
        p3 = np.array([point3.x, point3.y])
        
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle with improved numerical stability
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
        
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def smooth_keypoints(self, landmarks, person_id):
        """Step 6: Smooth Keypoints using moving average per person"""
        if person_id not in self.keypoint_history:
            # Initialize history for each landmark for this person
            self.keypoint_history[person_id] = {}
            for i, landmark in enumerate(landmarks.landmark):
                self.keypoint_history[person_id][i] = deque(maxlen=self.smoothing_window)
        
        # Add current landmarks to history
        for i, landmark in enumerate(landmarks.landmark):
            if i not in self.keypoint_history[person_id]:
                self.keypoint_history[person_id][i] = deque(maxlen=self.smoothing_window)
            self.keypoint_history[person_id][i].append([landmark.x, landmark.y, landmark.z])
        
        # Create smoothed landmarks
        smoothed_landmarks = []
        for i in range(len(landmarks.landmark)):
            if len(self.keypoint_history[person_id][i]) > 0:
                avg_x = sum([point[0] for point in self.keypoint_history[person_id][i]]) / len(self.keypoint_history[person_id][i])
                avg_y = sum([point[1] for point in self.keypoint_history[person_id][i]]) / len(self.keypoint_history[person_id][i])
                avg_z = sum([point[2] for point in self.keypoint_history[person_id][i]]) / len(self.keypoint_history[person_id][i])
                
                # Create a mock landmark object
                class MockLandmark:
                    def __init__(self, x, y, z):
                        self.x, self.y, self.z = x, y, z
                
                smoothed_landmarks.append(MockLandmark(avg_x, avg_y, avg_z))
            else:
                smoothed_landmarks.append(landmarks.landmark[i])
        
        return smoothed_landmarks
    
    def setup_frame_dimensions(self, frame):
        """Setup frame dimensions for MediaPipe optimization"""
        if self.frame_height is None or self.frame_width is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            print(f"Frame dimensions set: {self.frame_width}x{self.frame_height}")
    
    def process_frame(self, frame):
        """Process a single frame and return results"""
        # Setup dimensions on first frame
        self.setup_frame_dimensions(frame)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set writeable flag to False to improve performance
        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        return results
    
    def process_video(self, video_path, callback=None):
        """Step 2: Load Video and process"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video file")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {frame_count} frames at {fps:.2f} FPS")
        if callback: callback("status", f"Processing {frame_count} frames at {fps:.1f} FPS")
        
        frame_num = 0
        analysis_data = []
        
        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Process frame
            results = self.process_frame(frame)
            
            frame_data = {
                'frame_number': frame_num,
                'timestamp': frame_num / fps,
                'persons': []
            }
            
            if results.pose_landmarks:
                # For single person detection in video, we'll treat it as person 0
                person_id = 0
                
                # Step 4: Extract Keypoints of Interest
                # Step 6: Smooth Keypoints
                smoothed_landmarks = self.smooth_keypoints(results.pose_landmarks, person_id)
                
                # Step 7: Frame-Wise Feedback Logic
                feedback_data = self.feedback_analyzer.analyze_exercise_form(
                    smoothed_landmarks, 
                    self.current_exercise
                )
                
                # Step 9: Store Time-Series Data
                if person_id not in self.pose_data:
                    self.pose_data[person_id] = []
                
                frame_person_data = {
                    'person_id': person_id,
                    'landmarks': [(lm.x, lm.y, lm.z) for lm in smoothed_landmarks],
                    'feedback': feedback_data
                }
                
                self.pose_data[person_id].append(frame_person_data)
                frame_data['persons'].append(frame_person_data)
                
                # Step 8: Show Feedback
                person_detections = {person_id: feedback_data}
                frame = self.feedback_analyzer.draw_feedback(
                    frame, 
                    person_detections, 
                    frame_num,
                    self.current_exercise,
                    self.person_colors
                )
                
                # Draw pose landmarks with person color
                color = self.person_colors[person_id % len(self.person_colors)]
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2)
                )
            analysis_data.append(frame_data)
            
            # Update UI if callback provided
            if callback:
                # Convert frame for display
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                callback("frame", (display_frame, frame_data))
                
            # Show progress
            if frame_num % 30 == 0:  # Every 30 frames
                progress = (frame_num / frame_count) * 100
                print(f"Progress: {progress:.1f}%")
                if callback: callback("progress", progress)
            
            
        # Step 10: End and Release Resources
        cap.release()
        
        # Save analysis data automatically when video completes
        output_files = self.save_video_analysis(video_path, analysis_data)
        print(f"Processed {frame_num} frames")
        if callback: 
            callback("complete", {
                "frames_processed": frame_num,
                "output_files": output_files
            })
        return output_files
        
    
    def save_video_analysis(self, video_path, analysis_data):
        """Save video analysis data to JSON and CSV"""
        import os
        import pandas as pd
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_files = {}
        
        # Save JSON
        json_file = f"{base_name}_{timestamp}_analysis.json"
        with open(json_file, 'w') as f:
            json.dump({
                'video_path': video_path,
                'exercise': self.current_exercise,
                'analysis_data': analysis_data
            }, f, indent=2)
        output_files['json'] = json_file
        
        # Save CSV (flattened data)
        csv_data = []
        for frame in analysis_data:
            for person in frame['persons']:
                csv_data.append({
                    'frame_number': frame['frame_number'],
                    'timestamp': frame['timestamp'],
                    'person_id': person['person_id'],
                    'overall_score': person['feedback']['overall_score'],
                    'feedback': "; ".join(person['feedback']['feedback'])
                })
        
        if csv_data:
            csv_file = f"{base_name}_{timestamp}_analysis.csv"
            pd.DataFrame(csv_data).to_csv(csv_file, index=False)
            output_files['csv'] = csv_file
        
        print(f"Analysis data saved to {json_file} and {csv_file}")
        return output_files
    
    def process_webcam(self):
        """Step 2: Load Camera Feed and process multiple people"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set webcam properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increased for better multi-person detection
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting webcam... Press 'q' to quit")
        print("Now supporting true multi-person tracking and analysis!")
        
        frame_count = 0
        start_time = time.time()
        
        # Initialize variables for multi-person processing
        prev_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            results = self.process_frame(frame)
            
            person_detections = {}
            
            if results.pose_landmarks:
                # For webcam demo, we'll process each detected pose
                # Convert single pose detection to list format for consistency
                all_poses = [results.pose_landmarks]
                
                # Get center points for all detected poses
                center_points = [self._get_pose_center(pose.landmark) for pose in all_poses]
                
                # Update person tracker with new detections
                person_ids = self.person_tracker.update_tracks(center_points, frame_count)
                
                # Process each detected person
                for person_id, pose_landmarks in zip(person_ids, all_poses):
                    # Step 4: Extract Keypoints of Interest
                    # Step 6: Smooth Keypoints
                    smoothed_landmarks = self.smooth_keypoints(pose_landmarks, person_id)
                    
                    # Step 7: Frame-Wise Feedback Logic
                    feedback_data = self.feedback_analyzer.analyze_exercise_form(
                        smoothed_landmarks, 
                        self.current_exercise
                    )
                    person_detections[person_id] = feedback_data
                    
                    # Step 9: Store Time-Series Data
                    if person_id not in self.pose_data:
                        self.pose_data[person_id] = []
                    
                    self.pose_data[person_id].append({
                        'frame': frame_count,
                        'timestamp': time.time(),
                        'person_id': person_id,
                        'landmarks': [(lm.x, lm.y, lm.z) for lm in smoothed_landmarks],
                        'feedback': feedback_data
                    })
                    
                    # Draw pose landmarks with person color
                    color = self.person_colors[person_id % len(self.person_colors)]
                    self.mp_drawing.draw_landmarks(
                        frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2)
                    )
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Step 8: Show Feedback
            frame = self.feedback_analyzer.draw_feedback(
                frame, 
                person_detections, 
                frame_count,
                self.current_exercise,
                self.person_colors
            )
            
            # Display FPS on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show FPS in console periodically
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                print(f"FPS: {fps:.1f}, Avg FPS: {avg_fps:.1f}, Active persons: {len(person_detections)}")
            
            cv2.imshow('Multi-Person Exercise Form Analysis - Webcam', frame)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        # Step 10: End and Release Resources
        cap.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed
        print(f"Session complete: {frame_count} frames, average FPS: {avg_fps:.1f}")
        print(f"Total unique persons tracked: {len(self.pose_data)}")
    
    def _get_pose_center(self, landmarks):
        """Get center point of pose for tracking"""
        # Use torso center (average of shoulders and hips)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
        center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        
        return (center_x, center_y)
    
    def save_analysis_data(self, filename="multi_person_exercise_analysis.json"):
        """Step 9: Save time-series data for analysis"""
        try:
            # Convert data to serializable format
            serializable_data = {}
            for person_id, frames in self.pose_data.items():
                serializable_data[f"person_{person_id}"] = frames
            
            with open(filename, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            print(f"Multi-person analysis data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def generate_summary_report(self):
        """Generate summary statistics from collected data for all persons"""
        if not self.pose_data:
            print("No data to analyze")
            return
        
        print(f"\n=== Multi-Person Exercise Analysis Summary ===")
        print(f"Exercise: {self.feedback_analyzer.exercise_rules[self.current_exercise]['name']}")
        print(f"Total unique persons tracked: {len(self.pose_data)}")
        
        overall_stats = []
        
        for person_id, frames in self.pose_data.items():
            scores = [frame['feedback']['overall_score'] for frame in frames if 'feedback' in frame]
            if not scores:
                continue
            
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            print(f"\n--- Person {person_id + 1} ---")
            print(f"Frames analyzed: {len(frames)}")
            print(f"Average score: {avg_score:.2f}")
            print(f"Best score: {max_score:.2f}")
            print(f"Lowest score: {min_score:.2f}")
            
            # Performance categories
            excellent = sum(1 for s in scores if s >= 0.8)
            good = sum(1 for s in scores if 0.6 <= s < 0.8)
            needs_improvement = sum(1 for s in scores if s < 0.6)
            
            print(f"Performance breakdown:")
            print(f"  Excellent (≥0.8): {excellent} frames ({excellent/len(scores)*100:.1f}%)")
            print(f"  Good (0.6-0.8): {good} frames ({good/len(scores)*100:.1f}%)")
            print(f"  Needs improvement (<0.6): {needs_improvement} frames ({needs_improvement/len(scores)*100:.1f}%)")
            
            overall_stats.append({
                'person_id': person_id,
                'avg_score': avg_score,
                'frames': len(frames)
            })
        
        # Overall group statistics
        if overall_stats:
            group_avg = sum(stat['avg_score'] for stat in overall_stats) / len(overall_stats)
            best_performer = max(overall_stats, key=lambda x: x['avg_score'])
            most_active = max(overall_stats, key=lambda x: x['frames'])
            
            print(f"\n--- Group Statistics ---")
            print(f"Group average score: {group_avg:.2f}")
            print(f"Best performer: Person {best_performer['person_id'] + 1} (avg: {best_performer['avg_score']:.2f})")
            print(f"Most active: Person {most_active['person_id'] + 1} ({most_active['frames']} frames)")


class PersonTracker:
    """Enhanced person tracker for maintaining consistent IDs across frames"""
    
    def __init__(self, max_distance_threshold=0.3, max_frames_lost=10):
        self.tracks = {}  # {track_id: {'center': (x, y), 'last_seen': frame_num}}
        self.next_id = 0
        self.max_distance_threshold = max_distance_threshold
        self.max_frames_lost = max_frames_lost
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update_tracks(self, detections, frame_num):
        """Update tracks with new detections and return track IDs"""
        if not detections:
            # Remove old tracks
            self._cleanup_old_tracks(frame_num)
            return []
        
        track_ids = []
        
        # Convert detections to numpy array for distance calculation
        detection_points = np.array(detections)
        
        if not self.tracks:
            # First detections - create new tracks
            for detection in detections:
                self.tracks[self.next_id] = {
                    'center': detection,
                    'last_seen': frame_num
                }
                track_ids.append(self.next_id)
                self.next_id += 1
        else:
            # Match detections to existing tracks
            existing_centers = []
            existing_ids = []
            
            for track_id, track_data in self.tracks.items():
                existing_centers.append(track_data['center'])
                existing_ids.append(track_id)
            
            if existing_centers:
                existing_points = np.array(existing_centers)
                
                # Calculate distance matrix
                distances = cdist(detection_points, existing_points)
                
                # Hungarian algorithm would be ideal here, but for simplicity:
                # Match each detection to closest track within threshold
                used_tracks = set()
                
                for i, detection in enumerate(detections):
                    min_dist_idx = np.argmin(distances[i])
                    min_distance = distances[i][min_dist_idx]
                    closest_track_id = existing_ids[min_dist_idx]
                    
                    if min_distance < self.max_distance_threshold and closest_track_id not in used_tracks:
                        # Update existing track
                        self.tracks[closest_track_id]['center'] = detection
                        self.tracks[closest_track_id]['last_seen'] = frame_num
                        track_ids.append(closest_track_id)
                        used_tracks.add(closest_track_id)
                    else:
                        # Create new track
                        self.tracks[self.next_id] = {
                            'center': detection,
                            'last_seen': frame_num
                        }
                        track_ids.append(self.next_id)
                        self.next_id += 1
        
        # Clean up old tracks
        self._cleanup_old_tracks(frame_num)
        
        return track_ids
    
    def _cleanup_old_tracks(self, current_frame):
        """Remove tracks that haven't been seen for too long"""
        tracks_to_remove = []
        
        for track_id, track_data in self.tracks.items():
            if current_frame - track_data['last_seen'] > self.max_frames_lost:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]


if __name__ == "__main__":
    from exercise_analyzer_ui import ExerciseAnalyzerUI
    analyzer = MultiPersonExerciseFormAnalyzer()
    ui = ExerciseAnalyzerUI(analyzer)
    ui.run()