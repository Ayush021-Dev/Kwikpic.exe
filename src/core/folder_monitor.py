import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Callable, List
import threading

class PhotoFolderHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[str], None], deletion_callback: Callable[[str], None], supported_extensions: List[str] = None):
        super().__init__()
        self.callback = callback
        self.deletion_callback = deletion_callback
        self.supported_extensions = supported_extensions or ['.jpg', '.jpeg', '.png']
        self.processing_lock = threading.Lock()
        self.processing_queue = set()
        
    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            if any(file_path.lower().endswith(ext) for ext in self.supported_extensions):
                with self.processing_lock:
                    self.processing_queue.add(file_path)
                self.callback(file_path)
                
    def on_deleted(self, event):
        if not event.is_directory:
            file_path = event.src_path
            if any(file_path.lower().endswith(ext) for ext in self.supported_extensions):
                self.deletion_callback(file_path)

class FolderMonitor:
    def __init__(self, folder_path: str, callback: Callable[[str], None], deletion_callback: Callable[[str], None]):
        self.folder_path = folder_path
        self.callback = callback
        self.deletion_callback = deletion_callback
        self.observer = None
        self.handler = None
        self.is_running = False
        
    def start(self):
        if not self.is_running:
            self.handler = PhotoFolderHandler(self.callback, self.deletion_callback)
            self.observer = Observer()
            self.observer.schedule(self.handler, self.folder_path, recursive=True)
            self.observer.start()
            self.is_running = True
            
    def stop(self):
        if self.is_running and self.observer:
            self.observer.stop()
            self.observer.join()
            self.is_running = False
            
    def is_active(self) -> bool:
        return self.is_running 