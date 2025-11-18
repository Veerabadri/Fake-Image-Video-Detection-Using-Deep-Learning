# Fake Image & Video Detection Using Deep Learning

A Flask-based web application for detecting deepfake images and videos using Convolutional Neural Networks (CNN).

## Features

- **Image Detection**: Upload and analyze images for deepfake detection
- **Video Detection**: Upload and analyze videos frame-by-frame for deepfake detection
- **User Authentication**: Sign up and sign in functionality
- **Advanced Face Detection**: Uses MTCNN, dlib, and Haar Cascade for robust face detection
- **Probability-based Classification**: Enhanced detection logic with tunable thresholds

## Technology Stack

- **Backend**: Flask (Python)
- **Deep Learning**: TensorFlow/Keras
- **Face Detection**: MTCNN, dlib, Haar Cascade
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Database**: SQLite

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Veerabadri/Fake-Image-Video-Detection-Using-Deep-Learning.git
cd Fake-Image-Video-Detection-Using-Deep-Learning
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. **Important**: Model files (`models/*.h5`) are not included in this repository due to GitHub's file size limits. You need to:
   - Train your own models, or
   - Download pre-trained models and place them in the `models/` directory
   - Required models: `cnn.h5` (and optionally `Xception.h5`, `vgg19.h5`, `Inceptionresnet_v2.h5`)

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
.
├── app.py                 # Main Flask application
├── models/               # Model files (not in repo - see Installation)
├── static/               # Static files (CSS, JS, images)
├── templates/            # HTML templates
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Usage

1. **Sign Up/Sign In**: Create an account or log in
2. **Image Detection**: 
   - Navigate to the image detection page
   - Upload an image (PNG, JPG, JPEG)
   - View the detection result
3. **Video Detection**:
   - Navigate to the video detection page
   - Upload a video (MP4, AVI, MKV)
   - View the detection result

## Detection Logic

The application uses a sophisticated detection algorithm:
- **Face Extraction**: Automatically detects and extracts faces from images/videos
- **Probability Thresholding**: Uses configurable thresholds for classification
- **Frame Analysis**: For videos, analyzes multiple frames uniformly sampled across the video
- **Uncertainty Handling**: Flags suspicious overconfidence and uses variance analysis

## Configuration

Key parameters in `app.py`:
- `FAKE_THRESHOLD = 0.50`: Average fake probability threshold
- `FAKE_FRAME_THRESHOLD = 0.30`: Individual frame fake probability threshold
- `SUSPICIOUS_CONFIDENCE = 0.99`: Flags overconfident predictions

## Notes

- Dataset folders (`data/`, `Dataset-2/`, `Dataset-3/`) are excluded from the repository
- Model files are excluded due to GitHub's 100MB file size limit
- Large video files in `static/uploads/` are also excluded

## License

This project is for educational purposes.

## Author

Veerabadri

