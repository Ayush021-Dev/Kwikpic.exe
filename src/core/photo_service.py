import os
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .face_detector import FaceDetector, FaceLocation

class PhotoEventHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and self._is_image(event.src_path):
            self.service.process_new_photo(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and self._is_image(event.src_path):
            self.service.handle_deleted_photo(event.src_path)

    def _is_image(self, path: str) -> bool:
        return path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

class PhotoService:
    def __init__(self, watch_folder: str, output_folder: str):
        self.watch_folder = Path(watch_folder)
        self.output_folder = Path(output_folder)
        self.face_detector = FaceDetector()
        self.face_clusters: Dict[str, List[str]] = {}
        self.person_names: Dict[str, str] = {}
        self.observer = Observer()
        self.event_handler = PhotoEventHandler(self)
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start monitoring the folder"""
        self.observer.schedule(self.event_handler, str(self.watch_folder), recursive=False)
        self.observer.start()
        
        # Process existing photos
        self._process_existing_photos()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()

    def _process_existing_photos(self):
        """Process all existing photos in the watch folder"""
        for photo_path in self.watch_folder.glob('*'):
            if photo_path.is_file() and self._is_image(photo_path):
                self.process_new_photo(str(photo_path))

    def process_new_photo(self, photo_path: str):
        """Process a new photo and add it to appropriate cluster"""
        try:
            faces = self.face_detector.detect_faces(photo_path)
            if not faces:
                return

            # For simplicity, we'll use the first face found
            face = faces[0]
            
            # Find matching cluster or create new one
            matched_cluster = None
            for cluster_id, cluster_faces in self.face_clusters.items():
                if any(self.face_detector.compare_faces(face.encoding, self.face_detector.detect_faces(f)[0].encoding) < 0.6 
                      for f in cluster_faces):
                    matched_cluster = cluster_id
                    break

            if matched_cluster is None:
                # Create new cluster
                cluster_id = f"person_{len(self.face_clusters)}"
                self.face_clusters[cluster_id] = []
                self.person_names[cluster_id] = cluster_id

            # Add photo to cluster
            self.face_clusters[matched_cluster or cluster_id].append(photo_path)
            
            # Copy photo to output folder with cluster name
            output_name = f"{self.person_names[matched_cluster or cluster_id]}_{len(self.face_clusters[matched_cluster or cluster_id])}{Path(photo_path).suffix}"
            shutil.copy2(photo_path, self.output_folder / output_name)

        except Exception as e:
            print(f"Error processing photo {photo_path}: {e}")

    def handle_deleted_photo(self, photo_path: str):
        """Handle deleted photo by updating clusters"""
        for cluster_id, photos in self.face_clusters.items():
            if photo_path in photos:
                photos.remove(photo_path)
                if not photos:
                    del self.face_clusters[cluster_id]
                    del self.person_names[cluster_id]
                break

    def rename_person(self, cluster_id: str, new_name: str):
        """Rename a person cluster"""
        if cluster_id in self.person_names:
            self.person_names[cluster_id] = new_name
            # Rename all files in the output folder
            for i, photo_path in enumerate(self.face_clusters[cluster_id], 1):
                old_name = f"{cluster_id}_{i}{Path(photo_path).suffix}"
                new_name = f"{new_name}_{i}{Path(photo_path).suffix}"
                old_path = self.output_folder / old_name
                new_path = self.output_folder / new_name
                if old_path.exists():
                    old_path.rename(new_path)

    def _is_image(self, path: Path) -> bool:
        return path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.gif') 