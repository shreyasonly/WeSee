# """
# Real-Time Video Face Recognition System
# =======================================

# This system provides real-time face recognition using the device camera.
# It can identify registered people and display their names on screen.
# """

# import cv2
# import os
# import time
# import numpy as np
# import mediapipe as mp
# from datetime import datetime, date
# import threading
# import csv

# class RealTimeVideoRecognition:
#     """
#     Real-time video face recognition system.
#     """
    
#     def __init__(self, known_faces_dir="known_faces"):
#         """
#         Initialize the video recognition system.
        
#         Args:
#             known_faces_dir (str): Directory containing registered faces
#         """
#         self.known_faces_dir = known_faces_dir
#         self.cap = None
#         self.is_running = False
#         self.known_faces = {}
#         self.known_names = []
        
#         # Initialize MediaPipe
#         self.mp_face_detection = mp.solutions.face_detection
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.mp_drawing = mp.solutions.drawing_utils
        
#         # Load known faces
#         self.load_known_faces()
        
#         self.attendance_today = set()  # Track users marked present today
#         self.attendance_file = f"attendance_{date.today().isoformat()}.csv"
#         self._init_attendance_file()
#         self.last_attendance_overlay = {}  # name: timestamp for overlay
    
#     def load_known_faces(self):
#         """
#         Load all registered faces from the known_faces directory.
#         """
#         self.known_faces = {}
#         self.known_names = []
#         if not os.path.exists(self.known_faces_dir):
#             print(f"Known faces directory '{self.known_faces_dir}' not found.")
#             return
#         print("Loading registered faces...")
#         # Initialize face detection and mesh
#         with self.mp_face_detection.FaceDetection(
#             model_selection=1, min_detection_confidence=0.5
#         ) as face_detection:
#             with self.mp_face_mesh.FaceMesh(
#                 static_image_mode=True,
#                 max_num_faces=1,
#                 refine_landmarks=True,
#                 min_detection_confidence=0.5
#             ) as face_mesh:
#                 for filename in os.listdir(self.known_faces_dir):
#                     if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#                         name = os.path.splitext(filename)[0]
#                         # Remove _main or _fallback suffix if present
#                         if name.endswith('_main'):
#                             name = name[:-5]
#                         elif name.endswith('_fallback'):
#                             name = name[:-9]
#                         filepath = os.path.join(self.known_faces_dir, filename)
#                         try:
#                             image = cv2.imread(filepath)
#                             if image is None:
#                                 print(f"Could not read image: {filename}")
#                                 continue
#                             rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                             detection_results = face_detection.process(rgb_image)
#                             if detection_results.detections:
#                                 mesh_results = face_mesh.process(rgb_image)
#                                 if mesh_results.multi_face_landmarks:
#                                     landmarks = mesh_results.multi_face_landmarks[0]
#                                     embedding = self.extract_face_embedding(landmarks)
#                                     if embedding is not None:
#                                         self.known_faces[name] = embedding
#                                         self.known_names.append(name)
#                                         print(f"Loaded face: {name}")
#                                     else:
#                                         print(f"Warning: No embedding for {filename}")
#                                 else:
#                                     print(f"Warning: No face landmarks for {filename}")
#                             else:
#                                 print(f"Warning: No face detected in {filename}")
#                         except Exception as e:
#                             print(f"Error loading {filename}: {str(e)}")
#         print(f"Loaded {len(self.known_faces)} registered faces: {', '.join(self.known_names)}")
    
#     def extract_face_embedding(self, landmarks):
#         """
#         Extract face embedding from landmarks.
        
#         Args:
#             landmarks: MediaPipe face landmarks
            
#         Returns:
#             np.ndarray: Face embedding vector
#         """
#         try:
#             # Extract key facial points (simplified embedding)
#             points = []
            
#             # Key facial landmarks (eyes, nose, mouth corners)
#             key_indices = [33, 133, 362, 263, 61, 291, 199, 419]  # Simplified set
            
#             for idx in key_indices:
#                 if idx < len(landmarks.landmark):
#                     point = landmarks.landmark[idx]
#                     points.extend([point.x, point.y, point.z])
            
#             # Pad or truncate to fixed size
#             target_size = 24  # 8 points * 3 coordinates
#             if len(points) < target_size:
#                 points.extend([0] * (target_size - len(points)))
#             else:
#                 points = points[:target_size]
            
#             return np.array(points, dtype=np.float32)
            
#         except Exception as e:
#             print(f"Error extracting embedding: {str(e)}")
#             return None
    
#     def compare_faces(self, embedding1, embedding2, tolerance=0.6):
#         """
#         Compare two face embeddings.
        
#         Args:
#             embedding1 (np.ndarray): First face embedding
#             embedding2 (np.ndarray): Second face embedding
#             tolerance (float): Matching tolerance
            
#         Returns:
#             bool: True if faces match
#         """
#         try:
#             # Calculate cosine similarity
#             dot_product = np.dot(embedding1, embedding2)
#             norm1 = np.linalg.norm(embedding1)
#             norm2 = np.linalg.norm(embedding2)
            
#             if norm1 == 0 or norm2 == 0:
#                 return False
            
#             similarity = dot_product / (norm1 * norm2)
#             return similarity > tolerance
            
#         except Exception as e:
#             print(f"Error comparing faces: {str(e)}")
#             return False
    
#     def start_camera(self):
#         """
#         Start the camera for video capture.
        
#         Returns:
#             bool: True if camera started successfully
#         """
#         try:
#             self.cap = cv2.VideoCapture(0)
#             if not self.cap.isOpened():
#                 print("Error: Could not open camera.")
#                 return False
            
#             # Set camera properties
#             self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#             self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#             self.cap.set(cv2.CAP_PROP_FPS, 30)
            
#             print("Camera started successfully.")
#             return True
            
#         except Exception as e:
#             print(f"Error starting camera: {str(e)}")
#             return False
    
#     def stop_camera(self):
#         """
#         Stop the camera and release resources.
#         """
#         if self.cap:
#             self.cap.release()
#             self.cap = None
#         cv2.destroyAllWindows()
#         print("Camera stopped.")
    
#     def _init_attendance_file(self):
#         """Create the attendance file for today if it doesn't exist."""
#         if not os.path.exists(self.attendance_file):
#             with open(self.attendance_file, mode='w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["Name", "Date", "Time"])

#     def _mark_attendance(self, name):
#         """Mark attendance for a user if not already marked today."""
#         if name in self.attendance_today:
#             return  # Already marked today
#         now = datetime.now()
#         with open(self.attendance_file, mode='a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([name, now.date().isoformat(), now.strftime("%H:%M:%S")])
#         self.attendance_today.add(name)
#         self.last_attendance_overlay[name] = time.time()
    
#     def process_frame(self, frame):
#         """Process a frame, recognize faces, and mark attendance with overlay."""
#         frame = cv2.flip(frame, 1)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_boxes = []
#         face_embeddings = []
#         # Detect faces
#         with self.mp_face_detection.FaceDetection(
#             model_selection=1, min_detection_confidence=0.5
#         ) as face_detection:
#             detection_results = face_detection.process(rgb_frame)
#             if detection_results.detections:
#                 ih, iw, _ = frame.shape
#                 for detection in detection_results.detections:
#                     bboxC = detection.location_data.relative_bounding_box
#                     x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
#                                 int(bboxC.width * iw), int(bboxC.height * ih)
#                     x = max(0, x)
#                     y = max(0, y)
#                     w = min(w, iw - x)
#                     h = min(h, ih - y)
#                     if w > 0 and h > 0:
#                         face_boxes.append((x, y, w, h))
#                         with self.mp_face_mesh.FaceMesh(
#                             static_image_mode=False,
#                             max_num_faces=1,
#                             refine_landmarks=True,
#                             min_detection_confidence=0.5
#                         ) as face_mesh:
#                             mesh_results = face_mesh.process(rgb_frame)
#                             if mesh_results.multi_face_landmarks:
#                                 landmarks = mesh_results.multi_face_landmarks[0]
#                                 embedding = self.extract_face_embedding(landmarks)
#                                 face_embeddings.append(embedding)
#                             else:
#                                 face_embeddings.append(None)
#         assigned_names = set()
#         for i, (box, embedding) in enumerate(zip(face_boxes, face_embeddings)):
#             x, y, w, h = box
#             recognized_name = "UFO"
#             best_confidence = 0.0
#             if embedding is not None and len(self.known_faces) > 0:
#                 similarities = []
#                 for name, known_embedding in self.known_faces.items():
#                     dot_product = np.dot(embedding, known_embedding)
#                     norm1 = np.linalg.norm(embedding)
#                     norm2 = np.linalg.norm(known_embedding)
#                     if norm1 == 0 or norm2 == 0:
#                         similarity = 0.0
#                     else:
#                         similarity = dot_product / (norm1 * norm2)
#                     similarities.append((name, similarity))
#                 similarities.sort(key=lambda x: x[1], reverse=True)
#                 for name, confidence in similarities:
#                     if name not in assigned_names and confidence > 0.6:
#                         recognized_name = name
#                         best_confidence = confidence
#                         assigned_names.add(name)
#                         # Attendance logic
#                         self._mark_attendance(name)
#                         break
#             # Draw bounding box and name
#             if recognized_name != "UFO":
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 label = f"{recognized_name} ({best_confidence:.1%})"
#                 label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
#                 cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), (0, 255, 0), -1)
#                 cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#                 timestamp = datetime.now().strftime("%H:%M:%S")
#                 cv2.putText(frame, f"Detected: {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                 # Attendance overlay
#                 if recognized_name in self.last_attendance_overlay:
#                     overlay_time = self.last_attendance_overlay[recognized_name]
#                     if time.time() - overlay_time < 2.5:
#                         overlay_text = f"Welcome, {recognized_name}! Attendance marked."
#                         cv2.putText(frame, overlay_text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
#             else:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#                 cv2.putText(frame, "UFO", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         return frame
    
#     def run(self):
#         """
#         Run the real-time video recognition system.
#         """
#         if not self.known_faces:
#             print("No registered faces found. Please register faces first.")
#             return
        
#         if not self.start_camera():
#             return
        
#         self.is_running = True
#         print("Real-time video recognition started. Press 'Q' to quit.")
#         print(f"Looking for: {', '.join(self.known_names)}")
        
#         while self.is_running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 print("Error reading frame from camera.")
#                 break
            
#             # Process frame
#             processed_frame = self.process_frame(frame)
            
#             # Display frame
#             cv2.imshow("Real-Time Face Recognition - Press 'Q' to quit", processed_frame)
            
#             # Check for key press
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q') or key == ord('Q'):
#                 break
        
#         # Cleanup
#         self.stop_camera()
#         self.is_running = False
#         print("Real-time video recognition stopped.")


# def main():
#     """
#     Main function to run the real-time video recognition system.
#     """
#     print("Starting Real-Time Video Face Recognition System...")
#     print("This system identifies registered people through live video.")
#     print("=" * 60)
    
#     # Create and run the video recognition system
#     video_recognition = RealTimeVideoRecognition()
#     video_recognition.run()


# if __name__ == "__main__":
#     main() 

"""
Real-Time Video Face Recognition System
=======================================

This system provides state-of-the-art real-time face recognition using DeepFace embeddings,
temporal smoothing with Kalman filter, multi-frame voting to eliminate fluttering,
and integration with face_embeddings.py for persistent embedding storage.
"""

import cv2
import os
import time
import numpy as np
import mediapipe as mp
from datetime import datetime, date
import csv
import threading
from deepface import DeepFace
from collections import deque
from scipy.spatial import distance as dist
from filterpy.kalman import KalmanFilter
from face_embeddings import load_embeddings, save_embeddings, get_best_match

class RealTimeVideoRecognition:
    """
    Real-time video face recognition system with advanced anti-fluttering and embedding persistence.
    """
    
    def __init__(self, known_faces_dir="known_faces", embeddings_file="embeddings.json"):
        """
        Initialize the video recognition system.
        
        Args:
            known_faces_dir (str): Directory containing registered faces
            embeddings_file (str): JSON file for storing embeddings
        """
        self.known_faces_dir = known_faces_dir
        self.embeddings_file = embeddings_file
        self.cap = None
        self.is_running = False
        self.known_faces = {}  # {name: [embedding]}
        self.known_names = []
        
        # Initialize MediaPipe for face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
        # Face tracking state
        self.face_tracks = {}  # {track_id: {embedding, bbox, name, confidence, history}}
        self.next_track_id = 0
        self.max_track_age = 10  # Frames to keep a track alive
        self.voting_window = 5  # Frames for majority voting
        
        # Load known faces
        self.load_known_faces()
        
        self.attendance_today = set()
        self.attendance_file = f"attendance_{date.today().isoformat()}.csv"
        self._init_attendance_file()
        self.last_attendance_overlay = {}
        self.frame_count = 0
    
    def load_known_faces(self):
        """
        Load known face embeddings from embeddings.json or compute them if not cached.
        """
        self.known_faces = {}
        self.known_names = []
        
        # Try to load cached embeddings
        if os.path.exists(self.embeddings_file):
            try:
                self.known_faces = load_embeddings(self.embeddings_file)
                self.known_names = list(self.known_faces.keys())
                print(f"Loaded {len(self.known_names)} embeddings from {self.embeddings_file}")
            except Exception as e:
                print(f"Error loading embeddings: {e}")
        
        # Check for new or updated faces in known_faces_dir
        if not os.path.exists(self.known_faces_dir):
            print(f"Known faces directory '{self.known_faces_dir}' not found.")
            return
        
        updated = False
        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                if name.endswith('_main'):
                    name = name[:-5]
                elif name.endswith('_fallback'):
                    name = name[:-9]
                if name not in self.known_faces:
                    filepath = os.path.join(self.known_faces_dir, filename)
                    try:
                        image = cv2.imread(filepath)
                        if image is None:
                            print(f"Could not read image: {filename}")
                            continue
                        embedding = DeepFace.represent(
                            img_path=image,
                            model_name="ArcFace",
                            enforce_detection=False,
                            detector_backend="opencv"
                        )[0]["embedding"]
                        embedding = np.array(embedding, dtype=np.float32)
                        self.known_faces[name] = [embedding]  # List for compatibility with face_embeddings
                        self.known_names.append(name)
                        updated = True
                        print(f"Computed embedding for: {name}")
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
        
        # Save updated embeddings
        if updated:
            try:
                save_embeddings(self.known_faces, self.embeddings_file)
                print(f"Saved updated embeddings to {self.embeddings_file}")
            except Exception as e:
                print(f"Error saving embeddings: {e}")
        
        print(f"Loaded {len(self.known_faces)} registered faces: {', '.join(self.known_names)}")
    
    def reload_known_faces(self):
        """
        Reload embeddings from embeddings.json and recompute for new faces.
        """
        self.load_known_faces()
        print("Known faces reloaded successfully")
    
    def init_kalman_filter(self):
        """
        Initialize Kalman filter for smoothing bounding box coordinates.
        """
        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        kf.P *= 10.0
        kf.Q *= 0.1
        kf.R *= 5.0
        return kf
    
    def compute_iou(self, bbox1, bbox2):
        """
        Compute Intersection over Union between two bounding boxes.
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area
        return inter_area / union_area if union_area > 0 else 0
    
    def start_camera(self):
        """
        Start the camera for video capture.
        """
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open camera.")
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 25)
            print("Camera started successfully.")
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera and release resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        print("Camera stopped.")
    
    def _init_attendance_file(self):
        """Create the attendance file for today if it doesn't exist."""
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "Time"])

    def _mark_attendance(self, name):
        """Mark attendance for a user if not already marked today."""
        if name in self.attendance_today:
            return
        now = datetime.now()
        with open(self.attendance_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, now.date().isoformat(), now.strftime("%H:%M:%S")])
        self.attendance_today.add(name)
        self.last_attendance_overlay[name] = time.time()
    
    def process_frame(self, frame):
        """Process a frame, recognize faces, and apply anti-fluttering techniques."""
        self.frame_count += 1
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_boxes = []
        face_embeddings = []
        
        # Process every other frame
        if self.frame_count % 2 == 0:
            detection_results = self.face_detection.process(rgb_frame)
            if detection_results.detections:
                ih, iw, _ = frame.shape
                for detection in detection_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                int(bboxC.width * iw), int(bboxC.height * ih)
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, iw - x)
                    h = min(h, ih - y)
                    if w > 0 and h > 0:
                        face_boxes.append((x, y, w, h))
                        try:
                            face_img = frame[y:y+h, x:x+w]
                            embedding = DeepFace.represent(
                                img_path=face_img,
                                model_name="ArcFace",
                                enforce_detection=False,
                                detector_backend="opencv"
                            )[0]["embedding"]
                            face_embeddings.append(np.array(embedding, dtype=np.float32))
                        except Exception as e:
                            print(f"Error computing embedding: {e}")
                            face_embeddings.append(None)
        
        # Update face tracks
        new_tracks = {}
        for box, embedding in zip(face_boxes, face_embeddings):
            if embedding is None:
                continue
            best_track_id = None
            best_iou = 0
            for track_id, track in self.face_tracks.items():
                iou = self.compute_iou(box, track['bbox'])
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_track_id = track_id
            if best_track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                kf = self.init_kalman_filter()
                kf.x[:4] = np.array([x, y, w, h], dtype=np.float32)
                new_tracks[track_id] = {
                    'kf': kf,
                    'bbox': box,
                    'embedding': embedding,
                    'name': "UFO",
                    'confidence': 0.0,
                    'history': deque(maxlen=self.voting_window),
                    'age': 0
                }
            else:
                track = self.face_tracks[best_track_id]
                track['kf'].update(np.array([x, y, w, h], dtype=np.float32))
                track['bbox'] = box
                track['embedding'] = embedding if embedding is not None else track['embedding']
                track['age'] = 0
                new_tracks[best_track_id] = track
        
        # Update Kalman predictions and recognize faces
        for track_id, track in new_tracks.items():
            track['kf'].predict()
            x, y, w, h = track['kf'].x[:4].astype(int)
            track['bbox'] = (max(0, x), max(0, y), max(0, w), max(0, h))
            if track['embedding'] is not None and self.known_faces:
                best_name, best_dist, best_conf = get_best_match(
                    input_embedding=track['embedding'],
                    known_embeddings_dict=self.known_faces,
                    threshold=0.6,
                    distance_metric='cosine',
                    use_mean_vector=False
                )
                track['history'].append(best_name if best_name else "UFO")
                if len(track['history']) >= self.voting_window:
                    name_counts = {}
                    for name in track['history']:
                        name_counts[name] = name_counts.get(name, 0) + 1
                    track['name'] = max(name_counts, key=name_counts.get)
                    track['confidence'] = best_conf if track['name'] != "UFO" else 0.0
                else:
                    track['name'] = best_name if best_name else "UFO"
                    track['confidence'] = best_conf
                if track['name'] != "UFO":
                    self._mark_attendance(track['name'])
        
        self.face_tracks = {k: v for k, v in new_tracks.items() if v['age'] < self.max_track_age}
        for track in self.face_tracks.values():
            track['age'] += 1
        
        # Draw annotations
        for track_id, track in self.face_tracks.items():
            x, y, w, h = track['bbox']
            name = track['name']
            conf = track['confidence']
            color = (0, 255, 0) if name != "UFO" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{name} ({conf:.1%})" if name != "UFO" else "UFO"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Detected: {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if name in self.last_attendance_overlay:
                overlay_time = self.last_attendance_overlay[name]
                if time.time() - overlay_time < 2.5:
                    overlay_text = f"Welcome, {name}! Attendance marked."
                    cv2.putText(frame, overlay_text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
        
        return frame
    
    def run(self):
        """
        Run the real-time video recognition system.
        """
        if not self.known_faces:
            print("No registered faces found. Please register faces first.")
            return
        
        if not self.start_camera():
            return
        
        self.is_running = True
        print("Real-time video recognition started. Press 'Q' to quit.")
        print(f"Looking for: {', '.join(self.known_names)}")
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame from camera.")
                break
            
            processed_frame = self.process_frame(frame)
            cv2.imshow("Real-Time Face Recognition - Press 'Q' to quit", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
        
        self.stop_camera()
        self.is_running = False
        print("Real-time video recognition stopped.")


def main():
    """
    Main function to run the real-time video recognition system.
    """
    print("Starting Real-Time Video Face Recognition System...")
    print("This system uses DeepFace with anti-fluttering and persistent embeddings.")
    print("=" * 60)
    
    video_recognition = RealTimeVideoRecognition()
    video_recognition.run()


if __name__ == "__main__":
    main()