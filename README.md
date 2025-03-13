# Face Recognition Registration and Login System (OpenCV + Django)

This project is a **Face Recognition-based Authentication System** built using **Django** and **OpenCV**. Users can **register** with their face data and later log in using face recognition.

---

## 🔧 Features
- **Face Registration: Users can register with their facial images.**
- **Face Recognition Login: Authenticates users using real-time face detection.**
- **Dataset Management: Saves user facial images for training.**
- **Face-based Login Authentication**
- **Django Admin Panel for User Management**
- **Haar Cascade Classifier: Uses OpenCV's ```haarcascade_frontalface_default.xml``` for face detection.`**

---

## 🛠 Requirements
- Python (3.x)
- Django
- OpenCV (`opencv-python`)
- NumPy
- A good IDE like **VS Code** or **PyCharm**
- SQLite
- Webcam (for face recognition)
- Git & Github

---

## 📦 Installation

### 1️⃣ Clone the Repository:
```sh
git clone https://github.com/2347253/Face-Recognition-Login-System.git
cd Face-Recognition-Login-System
```

### 2️⃣ Create a Virtual Environment:
```sh
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

### 2️⃣ Create a Virtual Environment:
```sh
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

### 3️⃣ Install Dependencies:
```sh
pip install django opencv-python numpy pillow djangorestframework
pip install face-recognition dlib
```

### 4️⃣ Apply Migrations: 
```sh
python manage.py makemigrations
python manage.py migrate
```

### 5️⃣ Run the Development Server: 
```sh
python managae.py runserver
```

### 🖼️ Face Detection & Recognition

The system:

- Captures face images during registration.
- Stores images in the dataset/ folder.
- Trains the face recognition model (trainer.yml).
- Recognizes faces during login using Haarcascade.
