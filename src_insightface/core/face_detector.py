import cv2
import numpy as np
import face_recognition
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class FaceLocation:
    top: int
    right: int
    bottom: int
    left: int
    encoding: Optional[np.ndarray] = None

class FaceDetector:
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        
    def detect_faces(self, image_path: str) -> List[FaceLocation]:
        """
        Detect faces in an image and return their locations and encodings.
        """
        # Load image
        image = face_recognition.load_image_file(image_path)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(image)
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Create FaceLocation objects
        faces = []
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            faces.append(FaceLocation(
                top=top,
                right=right,
                bottom=bottom,
                left=left,
                encoding=encoding
            ))
            
        return faces
    
    def is_blurry(self, image_path: str, threshold: float = 100.0) -> bool:
        """
        Check if an image is blurry using Laplacian variance.
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold
    
    def compare_faces(self, face1: np.ndarray, face2: np.ndarray) -> float:
        """
        Compare two face encodings and return similarity score.
        """
        return face_recognition.face_distance([face1], face2)[0]
    
    def extract_face_image(self, image_path: str, face_location: FaceLocation) -> np.ndarray:
        """
        Extract a face image from the original image using face location.
        """
        image = cv2.imread(image_path)
        face_image = image[
            face_location.top:face_location.bottom,
            face_location.left:face_location.right
        ]
        return face_image 