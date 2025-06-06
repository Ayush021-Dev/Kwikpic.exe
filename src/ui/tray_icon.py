from PyQt5.QtWidgets import QSystemTrayIcon, QMenu, QAction, QStyle, QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import os

class TrayIcon(QSystemTrayIcon):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        
        # Create custom icon
        try:
            # Try to use a custom icon from the icons folder
            icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'icons', 'app_icon.png')
            if os.path.exists(icon_path):
                icon = QIcon(icon_path)
            else:
                # Fallback to system icon
                icon = QApplication.style().standardIcon(QStyle.SP_ComputerIcon)
        except:
            # Fallback to system icon
            icon = QApplication.style().standardIcon(QStyle.SP_ComputerIcon)

        self.setIcon(icon)
        self.setToolTip("Face Organizer")
        
        # Create menu
        menu = QMenu()
        
        # Show/Hide action
        self.show_action = QAction("Show", menu)
        self.show_action.triggered.connect(self.main_window.show)
        menu.addAction(self.show_action)
        
        # Select Folder action
        select_folder_action = QAction("Select Folder", menu)
        select_folder_action.triggered.connect(self.main_window.select_folder)
        menu.addAction(select_folder_action)
        
        # Separator
        menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", menu)
        exit_action.triggered.connect(self.exit_application)  # Connect to our new method instead
        menu.addAction(exit_action)
        
        self.setContextMenu(menu)
        
        # Connect signals
        self.activated.connect(self.tray_icon_activated)
        
    def tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            if self.main_window.isVisible():
                self.main_window.hide()
            else:
                self.main_window.show()
                self.main_window.activateWindow()
                
    def exit_application(self):
        """Properly exit the application by stopping monitoring and closing the window"""
        # First stop monitoring if active
        if hasattr(self.main_window, 'folder_monitor') and self.main_window.folder_monitor:
            if self.main_window.folder_monitor.is_active():
                self.main_window.folder_monitor.stop()
        
        # Then quit the application
        QApplication.quit()