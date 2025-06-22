import math
import cv2
import numpy as np

class FeedbackAnalyzer:
    def __init__(self):
        self.exercise_rules = self._define_exercise_rules()
    
    def _define_exercise_rules(self):
        """Define Rules for Form Analysis"""
        return {
            'bicep_curl': {
                'name': 'Bicep Curl',
                'rules': [
                    {'name': 'elbow_angle', 'min': 30, 'max': 160, 'weight': 0.4},
                    {'name': 'shoulder_stability', 'threshold': 0.1, 'weight': 0.3},
                    {'name': 'wrist_alignment', 'threshold': 0.15, 'weight': 0.3}
                ]
            },
            'deadlift': {
                'name': 'Deadlift',
                'rules': [
                    {'name': 'back_angle', 'min': 15, 'max': 45, 'weight': 0.4},
                    {'name': 'knee_hip_sync', 'threshold': 0.2, 'weight': 0.3},
                    {'name': 'bar_path', 'threshold': 0.1, 'weight': 0.3}
                ]
            },
            'lateral_raise': {
                'name': 'Lateral Raise',
                'rules': [
                    {'name': 'arm_elevation', 'min': 70, 'max': 110, 'weight': 0.4},
                    {'name': 'shoulder_alignment', 'threshold': 0.1, 'weight': 0.3},
                    {'name': 'elbow_bend', 'min': 160, 'max': 180, 'weight': 0.3}
                ]
            },
            'push_up': {
                'name': 'Push-Up',
                'rules': [
                    {'name': 'body_alignment', 'threshold': 0.1, 'weight': 0.4},
                    {'name': 'elbow_angle', 'min': 45, 'max': 160, 'weight': 0.3},
                    {'name': 'depth_control', 'threshold': 0.15, 'weight': 0.3}
                ]
            },
            'pull_up': {
                'name': 'Pull-Up',
                'rules': [
                    {'name': 'elbow_angle', 'min': 30, 'max': 180, 'weight': 0.4},
                    {'name': 'shoulder_engagement', 'threshold': 0.1, 'weight': 0.3},
                    {'name': 'body_stability', 'threshold': 0.15, 'weight': 0.3}
                ]
            },
            'back_squat': {
                'name': 'Back Squat',
                'rules': [
                    {'name': 'knee_angle', 'min': 70, 'max': 180, 'weight': 0.4},
                    {'name': 'hip_depth', 'threshold': 0.1, 'weight': 0.3},
                    {'name': 'back_posture', 'min': 15, 'max': 45, 'weight': 0.3}
                ]
            },
            'romanian_deadlift': {
                'name': 'Romanian Deadlift',
                'rules': [
                    {'name': 'hip_hinge', 'min': 20, 'max': 60, 'weight': 0.4},
                    {'name': 'knee_stability', 'threshold': 0.1, 'weight': 0.3},
                    {'name': 'back_neutral', 'threshold': 0.15, 'weight': 0.3}
                ]
            },
            'hammer_curl': {
                'name': 'Hammer Curl',
                'rules': [
                    {'name': 'elbow_angle', 'min': 30, 'max': 160, 'weight': 0.4},
                    {'name': 'wrist_neutral', 'threshold': 0.1, 'weight': 0.3},
                    {'name': 'shoulder_stability', 'threshold': 0.15, 'weight': 0.3}
                ]
            },
            'chest_press': {
                'name': 'Chest Press',
                'rules': [
                    {'name': 'elbow_angle', 'min': 45, 'max': 160, 'weight': 0.4},
                    {'name': 'shoulder_retraction', 'threshold': 0.1, 'weight': 0.3},
                    {'name': 'bar_path', 'threshold': 0.15, 'weight': 0.3}
                ]
            },
            'seated_shoulder_press': {
                'name': 'Seated Shoulder Press',
                'rules': [
                    {'name': 'elbow_angle', 'min': 45, 'max': 180, 'weight': 0.4},
                    {'name': 'back_support', 'threshold': 0.1, 'weight': 0.3},
                    {'name': 'overhead_alignment', 'threshold': 0.15, 'weight': 0.3}
                ]
            }
        }
    
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
    
    def analyze_exercise_form(self, landmarks, exercise_name):
        """Frame-Wise Feedback Logic"""
        if not exercise_name or exercise_name not in self.exercise_rules:
            return {"overall_score": 0, "feedback": ["No exercise selected"]}
        
        exercise_config = self.exercise_rules[exercise_name]
        scores = {}
        feedback = []
        
        # Get key landmarks (using MediaPipe pose landmark indices)
        try:
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
        except IndexError:
            return {"overall_score": 0, "feedback": ["Could not detect all required landmarks"]}
        
        # Apply exercise-specific rules
        for rule in exercise_config['rules']:
            rule_name = rule['name']
            weight = rule['weight']
            score = 0
            
            if rule_name == 'elbow_angle':
                # Calculate elbow angles
                left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                avg_angle = (left_angle + right_angle) / 2
                
                if rule['min'] <= avg_angle <= rule['max']:
                    score = 1.0
                else:
                    score = max(0, 1 - abs(avg_angle - (rule['max'] + rule['min'])/2) / 50)
                
                feedback.append(f"Elbow angle: {avg_angle:.1f}° ({'Good' if score > 0.7 else 'Needs improvement'})")
            
            elif rule_name == 'shoulder_stability':
                # Check shoulder movement stability
                shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
                score = 1.0 if shoulder_diff < rule['threshold'] else max(0, 1 - shoulder_diff / rule['threshold'])
                feedback.append(f"Shoulder stability: {'Good' if score > 0.7 else 'Keep shoulders level'}")
            
            elif rule_name == 'wrist_alignment':
                # Check wrist alignment with elbows
                left_align = abs(left_wrist.x - left_elbow.x)
                right_align = abs(right_wrist.x - right_elbow.x)
                avg_align = (left_align + right_align) / 2
                score = 1.0 if avg_align < rule['threshold'] else max(0, 1 - avg_align / rule['threshold'])
                feedback.append(f"Wrist alignment: {'Good' if score > 0.7 else 'Align wrists with elbows'}")
            
            elif rule_name == 'back_angle':
                # Calculate back angle (simplified)
                back_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
                if rule['min'] <= back_angle <= rule['max']:
                    score = 1.0
                else:
                    score = max(0, 1 - abs(back_angle - (rule['max'] + rule['min'])/2) / 30)
                feedback.append(f"Back angle: {back_angle:.1f}° ({'Good' if score > 0.7 else 'Adjust back position'})")
            
            elif rule_name == 'knee_angle':
                # Calculate knee angles
                left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
                avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
                
                if rule['min'] <= avg_knee_angle <= rule['max']:
                    score = 1.0
                else:
                    score = max(0, 1 - abs(avg_knee_angle - (rule['max'] + rule['min'])/2) / 40)
                feedback.append(f"Knee angle: {avg_knee_angle:.1f}° ({'Good' if score > 0.7 else 'Adjust knee bend'})")
            
            elif rule_name == 'body_alignment':
                # Check if body is in straight line (for push-ups)
                shoulder_hip_ankle_angle = self.calculate_angle(left_shoulder, left_hip, left_ankle)
                score = 1.0 if 160 <= shoulder_hip_ankle_angle <= 200 else max(0, 1 - abs(shoulder_hip_ankle_angle - 180) / 40)
                feedback.append(f"Body alignment: {'Good' if score > 0.7 else 'Keep body straight'}")
            
            elif rule_name == 'arm_elevation':
                # For lateral raises - check arm elevation
                shoulder_elbow_angle = self.calculate_angle(left_hip, left_shoulder, left_elbow)
                if rule['min'] <= shoulder_elbow_angle <= rule['max']:
                    score = 1.0
                else:
                    score = max(0, 1 - abs(shoulder_elbow_angle - (rule['max'] + rule['min'])/2) / 30)
                feedback.append(f"Arm elevation: {shoulder_elbow_angle:.1f}° ({'Good' if score > 0.7 else 'Adjust arm height'})")
            
            # Add more rule implementations as needed...
            else:
                # Default scoring for other rules
                score = 0.8  # Placeholder
                feedback.append(f"{rule_name}: Monitoring")
            
            scores[rule_name] = score * weight
        
        # Calculate overall score
        overall_score = sum(scores.values())
        
        return {
            "overall_score": overall_score,
            "individual_scores": scores,
            "feedback": feedback,
            "exercise": exercise_config['name']
        }
    
    def get_feedback_color(self, score):
        """Get color based on score"""
        if score >= 0.8:
            return (0, 255, 0)  # Green
        elif score >= 0.6:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 0, 255)  # Red
    
    def draw_feedback(self, image, person_detections, frame_num, current_exercise, person_colors):
        """Show Feedback on image for multiple persons"""
        if current_exercise not in self.exercise_rules:
            return image
            
        height, width = image.shape[:2]
        
        # Draw exercise name at top
        cv2.putText(image, f"Exercise: {self.exercise_rules[current_exercise]['name']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw total people count
        cv2.putText(image, f"People detected: {len(person_detections)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw feedback for each person
        for i, (person_id, feedback_data) in enumerate(person_detections.items()):
            if feedback_data is None:
                continue
                
            color = person_colors[person_id % len(person_colors)]
            
            # Calculate position for this person's feedback
            x_offset = 10 + (i * 300)  # Horizontal spacing
            y_start = 100
            
            # Draw person ID and overall score
            score = feedback_data["overall_score"]
            score_color = self.get_feedback_color(score)
            
            cv2.putText(image, f"Person {person_id+1}: {score:.2f}", 
                       (x_offset, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw individual feedback (limit to first 3 items to save space)
            for j, feedback_msg in enumerate(feedback_data["feedback"][:3]):
                cv2.putText(image, feedback_msg, (x_offset, y_start + 25 + j * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return image