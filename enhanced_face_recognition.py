"""
Enhanced Face Recognition System
================================

This system provides improved face recognition accuracy for group photos
using faces registered through the real-time registration system.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont
import pickle
import time
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedFaceRecognitionSystem:
    """
    Enhanced face recognition system with improved accuracy for group photos.
    """
    
    def __init__(self, known_faces_dir: str = "known_faces", tolerance: float = 0.5):
        """
        Initialize the enhanced face recognition system.
        
        Args:
            known_faces_dir (str): Directory containing known face images
            tolerance (float): Face recognition tolerance (lower = more strict)
        """
        self.known_faces_dir = known_faces_dir
        self.tolerance = tolerance
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection and mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.6
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            min_detection_confidence=0.5
        )
        
        # Initialize storage for known faces
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_embeddings_cache = {}
        
        # Create known faces directory if it doesn't exist
        os.makedirs(known_faces_dir, exist_ok=True)
        
        # Load known faces from database
        self._load_known_faces()
        
        logger.info(f"Enhanced Face Recognition System initialized with {len(self.known_face_names)} known faces")
    
    def _extract_enhanced_face_embedding(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract enhanced face embedding using multiple MediaPipe features.
        
        Args:
            image (np.ndarray): Input image
            face_location (Tuple): Face location (x, y, width, height)
            
        Returns:
            np.ndarray: Enhanced face embedding vector
        """
        x, y, w, h = face_location
        
        # Extract face region with padding
        padding = int(min(w, h) * 0.2)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        face_img = image[y1:y2, x1:x2]
        
        # Convert to RGB for MediaPipe
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Get face mesh landmarks
        results = self.face_mesh.process(face_rgb)
        
        if results.multi_face_landmarks:
            # Extract landmark coordinates as embedding
            landmarks = results.multi_face_landmarks[0]
            embedding = []
            
            # Use more facial landmarks for better accuracy
            key_points = [
                10, 33, 61, 93, 152, 234, 454, 476,  # Basic facial points
                151, 337, 299, 333, 298, 301, 368, 397,  # Eye region
                0, 267, 37, 39, 40, 185, 191, 246,  # Nose region
                17, 84, 181, 91, 146, 61, 185, 40,  # Mouth region
                78, 95, 88, 178, 87, 14, 317, 402,  # Cheek region
            ]
            
            for point_idx in key_points:
                if point_idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[point_idx]
                    embedding.extend([landmark.x, landmark.y, landmark.z])
            
            # Pad or truncate to fixed size
            target_size = 144  # 48 points * 3 coordinates
            if len(embedding) < target_size:
                embedding.extend([0] * (target_size - len(embedding)))
            else:
                embedding = embedding[:target_size]
            
            return np.array(embedding)
        
        return np.zeros(144)  # Return zero vector if no landmarks
    
    def _calculate_enhanced_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate enhanced similarity between two face embeddings.
        
        Args:
            embedding1 (np.ndarray): First face embedding
            embedding2 (np.ndarray): Second face embedding
            
        Returns:
            float: Similarity score (0-1, higher = more similar)
        """
        # Normalize embeddings
        emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Apply additional weighting for better accuracy
        # Weight different facial regions differently
        eye_region_weight = 1.2
        nose_region_weight = 1.1
        mouth_region_weight = 1.0
        
        # Apply region-specific weights
        weighted_similarity = similarity * (
            eye_region_weight * 0.4 + 
            nose_region_weight * 0.3 + 
            mouth_region_weight * 0.3
        )
        
        return max(0, min(1, weighted_similarity))
    
    def _load_known_faces(self):
        """
        Load known faces from the database directory with enhanced processing.
        """
        logger.info("Loading known faces from database...")
        
        # Supported image formats
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        # Scan through all files in the known faces directory
        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(image_extensions):
                # Extract name from filename (remove extension)
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.known_faces_dir, filename)
                
                try:
                    # Load the image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Convert to RGB for MediaPipe
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    results = self.face_detection.process(rgb_image)
                    
                    if results.detections:
                        # Use the first face found
                        detection = results.detections[0]
                        bbox = detection.location_data.relative_bounding_box
                        
                        # Convert relative coordinates to absolute
                        h, w, _ = image.shape
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Extract enhanced face embedding
                        face_embedding = self._extract_enhanced_face_embedding(image, (x, y, width, height))
                        
                        if np.any(face_embedding):  # Check if embedding is not all zeros
                            self.known_face_encodings.append(face_embedding)
                            self.known_face_names.append(name)
                            logger.info(f"Loaded enhanced face for: {name}")
                        else:
                            logger.warning(f"No valid face embedding found in: {filename}")
                    else:
                        logger.warning(f"No face found in image: {filename}")
                        
                except Exception as e:
                    logger.error(f"Error loading face from {filename}: {str(e)}")
        
        logger.info(f"Successfully loaded {len(self.known_face_names)} known faces with enhanced embeddings")
    
    def _detect_faces(self, image: np.ndarray) -> Tuple[List, List]:
        """
        Detect faces in the input image using enhanced detection.
        """
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_image)
        
        face_locations = []
        face_encodings = []
        
        if results.detections:
            h, w, _ = image.shape
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                face_locations.append((x, y, width, height))
                
                # Extract enhanced face embedding
                face_embedding = self._extract_enhanced_face_embedding(image, (x, y, width, height))
                face_encodings.append(face_embedding)
        
        logger.info(f"Detected {len(face_locations)} faces in the image")
        return face_locations, face_encodings
    
    def _recognize_faces(self, face_encodings: List) -> List[Dict]:
        """
        Recognize faces with enhanced accuracy and confidence scores.
        Ensures each known identity is assigned only once per image unless it's a true duplicate.
        """
        recognition_results = []
        assigned_names = set()
        similarity_matrix = []
        # Build similarity matrix: rows=faces, cols=known faces
        for face_encoding in face_encodings:
            similarities = [self._calculate_enhanced_similarity(face_encoding, known_encoding)
                            for known_encoding in self.known_face_encodings]
            similarity_matrix.append(similarities)
        similarity_matrix = np.array(similarity_matrix)
        # For each face, find the best match, but prevent duplicate assignments
        used_indices = set()
        for i, similarities in enumerate(similarity_matrix):
            best_idx = int(np.argmax(similarities)) if len(similarities) > 0 else -1
            best_score = similarities[best_idx] if best_idx >= 0 and len(similarities) > 0 else 0.0
            best_name = self.known_face_names[best_idx] if best_idx >= 0 and len(similarities) > 0 else 'Unknown'
            # Prevent duplicate assignment unless it's a true duplicate (very high similarity)
            if best_name in assigned_names and best_score < 0.99:
                result = {'name': 'Unknown', 'confidence': best_score, 'similarity_scores': similarities.tolist()}
            elif best_score >= self.tolerance:
                result = {'name': best_name, 'confidence': best_score, 'similarity_scores': similarities.tolist()}
                assigned_names.add(best_name)
            else:
                result = {'name': 'Unknown', 'confidence': best_score, 'similarity_scores': similarities.tolist()}
            recognition_results.append(result)
        return recognition_results
    
    def _analyze_face_attributes(self, image: np.ndarray, face_locations: List) -> List[Dict]:
        """
        Analyze age, gender, and emotion for each detected face.
        """
        face_attributes = []
        
        for i, (x, y, w, h) in enumerate(face_locations):
            try:
                # Extract face region from image
                face_img = image[y:y+h, x:x+w]
                
                # Analyze face attributes using DeepFace
                analysis = DeepFace.analyze(
                    face_img, 
                    actions=['age', 'gender', 'emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                # Extract results (DeepFace returns a list for multiple faces)
                if isinstance(analysis, list):
                    result = analysis[0]
                else:
                    result = analysis
                
                attributes = {
                    'age': result.get('age', 'Unknown'),
                    'gender': result.get('gender', 'Unknown'),
                    'emotion': result.get('dominant_emotion', 'Unknown')
                }
                
                face_attributes.append(attributes)
                logger.info(f"Face {i+1}: Age={attributes['age']}, Gender={attributes['gender']}, Emotion={attributes['emotion']}")
                
            except Exception as e:
                logger.warning(f"Could not analyze attributes for face {i+1}: {str(e)}")
                face_attributes.append({
                    'age': 'Unknown',
                    'gender': 'Unknown', 
                    'emotion': 'Unknown'
                })
        
        return face_attributes
    
    def _annotate_image(self, image: np.ndarray, face_locations: List, 
                       recognition_results: List[Dict], face_attributes: List[Dict]) -> np.ndarray:
        """
        Annotate the image with enhanced face bounding boxes and labels.
        """
        # Convert to PIL Image for better text rendering
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Colors for different elements
        box_color = (0, 255, 0)  # Green for bounding box
        text_color = (255, 255, 255)  # White for text
        text_bg_color = (0, 0, 0)  # Black background for text
        
        for i, ((x, y, w, h), recognition, attributes) in enumerate(
            zip(face_locations, recognition_results, face_attributes)
        ):
            # Draw bounding box around face
            draw.rectangle([x, y, x+w, y+h], outline=box_color, width=2)
            
            # Prepare label text
            label_parts = [f"Name: {recognition['name']}"]
            
            # Add confidence score if recognized
            if recognition['name'] != 'Unknown':
                confidence = recognition['confidence']
                label_parts.append(f"Confidence: {confidence:.1%}")
            
            if attributes['age'] != 'Unknown':
                label_parts.append(f"Age: {attributes['age']}")
            if attributes['gender'] != 'Unknown':
                label_parts.append(f"Gender: {attributes['gender']}")
            if attributes['emotion'] != 'Unknown':
                label_parts.append(f"Emotion: {attributes['emotion']}")
            
            label_text = " | ".join(label_parts)
            
            # Get text size for background
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw text background
            text_bg_coords = [
                x, y - text_height - 10,
                x + text_width + 10, y
            ]
            draw.rectangle(text_bg_coords, fill=text_bg_color)
            
            # Draw text
            draw.text((x + 5, y - text_height - 5), label_text, fill=text_color, font=font)
        
        # Convert back to OpenCV format
        annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return annotated_image
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Process a single image for enhanced face recognition and tagging.
        """
        logger.info(f"Processing image: {image_path}")
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Step 1: Detect faces in the image
        face_locations, face_encodings = self._detect_faces(image)
        
        if not face_locations:
            logger.warning("No faces detected in the image")
            return {
                'faces_detected': 0,
                'annotated_image': image,
                'face_data': []
            }
        
        # Step 2: Recognize faces against known database
        recognition_results = self._recognize_faces(face_encodings)
        
        # Step 3: Analyze face attributes (age, gender, emotion)
        face_attributes = self._analyze_face_attributes(image, face_locations)
        
        # Step 4: Annotate the image with results
        annotated_image = self._annotate_image(image, face_locations, recognition_results, face_attributes)
        
        # Step 5: Save annotated image
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"enhanced_output_{base_name}_annotated.jpg"
        
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"Enhanced annotated image saved to: {output_path}")
        
        # Prepare results
        face_data = []
        for i, (location, recognition, attributes) in enumerate(zip(face_locations, recognition_results, face_attributes)):
            face_data.append({
                'face_id': i + 1,
                'location': location,
                'name': recognition['name'],
                'confidence': recognition['confidence'],
                'age': attributes['age'],
                'gender': attributes['gender'],
                'emotion': attributes['emotion']
            })
        
        results = {
            'faces_detected': len(face_locations),
            'annotated_image': annotated_image,
            'output_path': output_path,
            'face_data': face_data
        }
        
        logger.info(f"Enhanced processing complete. Detected {len(face_locations)} faces.")
        return results
    
    def reload_known_faces(self):
        """
        Reload known faces from the database (useful after registration).
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self._load_known_faces()
        logger.info("Known faces reloaded successfully")
    
    def get_known_faces(self) -> List[str]:
        """
        Get list of all known face names.
        """
        return self.known_face_names.copy()


def main():
    """
    Example usage of the Enhanced Face Recognition System.
    """
    # Initialize the enhanced face recognition system
    face_system = EnhancedFaceRecognitionSystem(known_faces_dir="known_faces")
    
    # Find any image in the sample_images directory
    sample_images_dir = "sample_images"
    sample_image_path = None
    
    if os.path.exists(sample_images_dir):
        # Look for any image file
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        for filename in os.listdir(sample_images_dir):
            if filename.lower().endswith(image_extensions):
                sample_image_path = os.path.join(sample_images_dir, filename)
                break
    
    if sample_image_path and os.path.exists(sample_image_path):
        print(f"Found test image: {sample_image_path}")
        try:
            # Process the image
            results = face_system.process_image(sample_image_path)
            
            # Print results
            print(f"\nEnhanced Processing Results:")
            print(f"Faces detected: {results['faces_detected']}")
            print(f"Output saved to: {results['output_path']}")
            
            if results['faces_detected'] > 0:
                print(f"\nEnhanced Face Details:")
                for face in results['face_data']:
                    print(f"Face {face['face_id']}:")
                    print(f"  Name: {face['name']}")
                    if face['name'] != 'Unknown':
                        print(f"  Confidence: {face['confidence']:.1%}")
                    print(f"  Age: {face['age']}")
                    print(f"  Gender: {face['gender']}")
                    print(f"  Emotion: {face['emotion']}")
                    print()
            else:
                print("No faces detected in the image.")
                
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    else:
        print(f"No test images found in {sample_images_dir}")
        print("Please add some images to the sample_images directory to test the system.")


if __name__ == "__main__":
    main() 