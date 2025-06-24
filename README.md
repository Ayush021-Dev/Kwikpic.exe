# 📸  Kwikpic- Face Organizer

**Kwikpic** is a smart desktop photo management tool that automatically organizes your image collections using face detection and recognition. Inspired by the already existing Kwikpic app which is made for android phones — fully offline and privacy-respecting — this app clusters photos by person, filters out blurry images, and continuously monitors folders for changes. It runs silently in the background and is packaged as a standalone `.exe` for easy use on Windows systems.

---

## ✨ Features

- 🔍 **Automatic Face Detection & Clustering**  
  Groups all photos by unique people using advanced face recognition.

- 📂 **Live Folder Monitoring**  
  Detects new photo additions or deletions in real time.

- 🧠 **Search by Face**  
  Scan your own face to find and display all related photos.

- 🧹 **Blurry Image Filtering**  
  Automatically filters out low-quality images.

- 📊 **Photo Insights**  
  Shows statistics such as number of unique people, most-photographed person, and duplicates.

- 🖱️ **Drag-and-Drop Import**  
  Easily import new photo sets with simple drag-and-drop.

- ⚙️ **Runs Silently in the Background**  
  Works like a Windows background service or startup app.

- 📦 **Packaged Executable (.exe)**  
  Built using Nuitka for standalone Windows distribution — no need to install Python or dependencies.

---

## 🛠️ System Requirements

- **OS**: Windows 10/11 (recommended), Linux/Mac supported for development
- **Python**: 3.8 or higher (for development)
- **RAM**: 4GB minimum recommended
- **Storage**: Depends on the number of images

---

## 📦 Installation (Development)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/face-organizer.git
cd face-organizer
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# Activate:
venv\Scripts\activate       # On Windows
source venv/bin/activate    # On Linux/macOS
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python main.py
```

---

## 🪄 Build as Windows Executable (.exe)

To package the app into a standalone `.exe` using **Nuitka**:

### Install Nuitka
```bash
pip install nuitka
```

### Run the Build Command
```bash
python -m nuitka main.py \
  --standalone \
  --onefile \
  --windows-console-mode=disable \
  --enable-plugin=pyqt5 \
  --output-dir=dist \
  --include-data-files=kwikpic/Lib/site-packages/face_recognition_models/models/*.dat=face_recognition_models/models/
```

> ✅ This creates a `.exe` in the `dist/` directory that can be used on any Windows machine.


## 🤖 Face Recognition Models

Ensure `face_recognition_models` are properly bundled. Nuitka includes these using:

```bash
--include-data-files=path/to/models/*.dat=face_recognition_models/models/
```

## 🤝 Contributing

Pull requests, issues, and suggestions are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes
4. Push to the branch
5. Open a pull request
