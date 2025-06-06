import sys
import os
import psutil
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from src.ui.main_window import MainWindow
from src.ui.tray_icon import TrayIcon

def set_process_priority_low():
    try:
        # Get the current process
        process = psutil.Process(os.getpid())
        # Set to low priority (BELOW_NORMAL_PRIORITY_CLASS in Windows)
        if sys.platform == 'win32':
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            # For Unix-like systems
            process.nice(10)
    except Exception as e:
        print(f"Failed to set process priority: {e}")

def main():
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Prevent app from quitting when window is closed
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create main window
    window = MainWindow()
    
    # Create system tray icon
    tray_icon = TrayIcon(window)
    tray_icon.show()
    
    # Set process priority to low
    set_process_priority_low()
    
    # Start minimized to tray if not the first run
    settings_file = os.path.join(os.path.expanduser('~'), '.face_organizer_settings')
    if os.path.exists(settings_file):
        # Not first run, start minimized
        pass
    else:
        # First run, show window and create settings file
        window.show()
        with open(settings_file, 'w') as f:
            f.write('initialized=true')
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
