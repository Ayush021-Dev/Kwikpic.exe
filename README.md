# Face Organizer

A professional photo organization application that uses facial recognition to automatically group and organize your photos.

## Features

- Automatic face detection and recognition
- Smart photo organization by person
- Modern, intuitive user interface
- Blurry image detection and filtering
- Drag-and-drop photo import
- Photo collection statistics and insights
- Customizable settings and preferences

## Requirements

- Python 3.8 or higher
- Dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
python main.py
```

## Project Structure

- `main.py` - Application entry point
- `src/` - Source code directory
  - `core/` - Core face detection and recognition engine
  - `data/` - Data management layer
  - `logic/` - Business logic layer
  - `ui/` - User interface components
  - `config/` - Configuration and settings
- `assets/` - Application resources (images, icons, etc.)
- `tests/` - Unit tests

## License

MIT License 