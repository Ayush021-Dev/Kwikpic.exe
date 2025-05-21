from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from .face_detector import FaceLocation, FaceDetector
from sklearn.cluster import DBSCAN
import os

@dataclass
class Person:
    id: int
    name: str
    face_encodings: List[np.ndarray]
    photo_paths: Set[str]
    face_indices: List[int]  # Indices of faces in the global list

class FaceRecognizer:
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold
        self.detector = FaceDetector()
        self.people: Dict[int, Person] = {}
        self.face_data: List[Tuple[str, FaceLocation]] = []  # (image_path, FaceLocation)
        self.cluster_labels: List[int] = []

    def scan_folder(self, folder_path: str, progress_callback=None):
        """
        Scan all images in the folder, detect faces, and cluster them using DBSCAN.
        """
        self.face_data.clear()
        self.people.clear()
        self.cluster_labels.clear()
        encodings = []
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total = len(image_files)
        for i, image_path in enumerate(image_files):
            faces = self.detector.detect_faces(image_path)
            for face in faces:
                if face.encoding is not None:
                    encodings.append(face.encoding)
                    self.face_data.append((image_path, face))
            if progress_callback is not None:
                progress_callback(int((i + 1) / total * 100))
        if not encodings:
            return
        encodings_np = np.stack(encodings)
        # DBSCAN clustering
        db = DBSCAN(eps=self.similarity_threshold, min_samples=1, metric='euclidean').fit(encodings_np)
        self.cluster_labels = db.labels_
        # Group faces by cluster
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(self.cluster_labels):
            clusters.setdefault(label, []).append(idx)
        self.people = {}
        for cluster_id, indices in clusters.items():
            photo_paths = set(self.face_data[i][0] for i in indices)
            encs = [encodings[i] for i in indices]
            self.people[cluster_id] = Person(
                id=cluster_id,
                name=f"Person {cluster_id}",
                face_encodings=encs,
                photo_paths=photo_paths,
                face_indices=indices
            )

    def get_all_people(self) -> List[Person]:
        return list(self.people.values())

    def get_person_photos(self, person_id: int) -> Set[str]:
        if person_id in self.people:
            return self.people[person_id].photo_paths
        return set()

    def get_person_face_indices(self, person_id: int) -> List[int]:
        if person_id in self.people:
            return self.people[person_id].face_indices
        return []

    def rename_person(self, person_id: int, new_name: str) -> None:
        if person_id in self.people:
            self.people[person_id].name = new_name 