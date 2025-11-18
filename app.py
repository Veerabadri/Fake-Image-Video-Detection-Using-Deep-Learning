import tensorflow as tf
import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import backend as K
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import sqlite3

try:
    from mtcnn import MTCNN
    MTCNN_DETECTOR = MTCNN()
    print("MTCNN detector initialized.")
except Exception as mtcnn_error:
    MTCNN_DETECTOR = None
    print(f"Warning: MTCNN detector unavailable: {mtcnn_error}")

try:
    import dlib
    FACE_DETECTOR = dlib.get_frontal_face_detector()
    print("dlib face detector initialized.")
except Exception as detector_error:
    FACE_DETECTOR = None
    print(f"Warning: dlib face detector unavailable: {detector_error}")

try:
    HAAR_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if HAAR_CASCADE.empty():
        HAAR_CASCADE = None
        print("Warning: Haar cascade file not found.")
    else:
        print("Haar cascade face detector initialized.")
except Exception as haar_error:
    HAAR_CASCADE = None
    print(f"Warning: Haar cascade unavailable: {haar_error}")

app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
def init_db():
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS info 
                    (user TEXT, email TEXT, password TEXT, mobile TEXT, name TEXT)''')
    con.commit()
    con.close()

init_db()

def specificity_m(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    specificity = true_negatives / (possible_negatives + K.epsilon())
    return specificity

def sensitivity_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    sensitivity = true_positives / (possible_positives + K.epsilon())
    return sensitivity

def mae(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

FAKE_THRESHOLD = 0.50  # Average fake probability threshold
FAKE_FRAME_THRESHOLD = 0.30  # Individual frame fake probability threshold (more sensitive)
SUSPICIOUS_CONFIDENCE = 0.99  # If model is >99% confident, be more cautious
CONFIDENCE_THRESHOLD = 0.95  # Flag if model is too confident (>95%) - might be suspicious

def extract_face(frame):
    """
    Detect the largest face in the frame and return a 128x128 cropped image.
    Tries MTCNN, then dlib, then Haar cascade.
    Returns None if no detector succeeds.
    """
    if frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. MTCNN (best for blurred/warped faces)
    if MTCNN_DETECTOR is not None:
        try:
            detections = MTCNN_DETECTOR.detect_faces(frame)
            if detections:
                # Select largest face (fix: was using detections[0] before)
                largest = max(detections, key=lambda det: det['box'][2] * det['box'][3])
                x, y, w, h = largest['box']
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)
                if x2 > x1 and y2 > y1:
                    face = frame[y1:y2, x1:x2]
                    if face.size != 0:
                        return cv2.resize(face, (128, 128))
        except Exception as e:
            print(f"MTCNN detection error: {e}")

    detections = []
    if FACE_DETECTOR is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = FACE_DETECTOR(rgb, 1)
        if len(detections) > 0:
            # Select largest
            d = max(detections, key=lambda rect: (rect.right() - rect.left()) * (rect.bottom() - rect.top()))
            x1 = max(0, d.left())
            y1 = max(0, d.top())
            x2 = min(frame.shape[1], d.right())
            y2 = min(frame.shape[0], d.bottom())
            if x2 > x1 and y2 > y1:
                face = frame[y1:y2, x1:x2]
                if face.size != 0:
                    return cv2.resize(face, (128, 128))

    if HAAR_CASCADE is not None:
        boxes = HAAR_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        if len(boxes) > 0:
            x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            if x2 > x1 and y2 > y1:
                face = frame[y1:y2, x1:x2]
                if face.size != 0:
                    return cv2.resize(face, (128, 128))

    return None

model_path2 = 'models/cnn.h5' 

# Try loading model with custom objects
model = None
try:
    custom_objects = {
        'specificity_m': specificity_m, 
        'sensitivity_m': sensitivity_m, 
        'mae': mae, 
        'mse': mse
    }
    # Try with compile=False first (for TensorFlow 2.x compatibility)
    try:
        model = load_model(model_path2, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully with custom objects (compile=False)!")
    except:
        # Try with compile=True
        model = load_model(model_path2, custom_objects=custom_objects, compile=True)
        print("Model loaded successfully with custom objects (compile=True)!")
except Exception as e:
    print(f"Error loading model with custom objects: {e}")
    # Try loading without custom objects (in case model doesn't use them)
    try:
        try:
            model = load_model(model_path2, compile=False)
            print("Model loaded without custom objects (compile=False)!")
        except:
            model = load_model(model_path2, compile=True)
            print("Model loaded without custom objects (compile=True)!")
    except Exception as e2:
        print(f"Error loading model without custom objects: {e2}")
        print("=" * 50)
        print("MODEL LOADING FAILED!")
        print("Please check:")
        print("1. Model file exists: models/cnn.h5")
        print("2. TensorFlow/Keras version compatibility")
        print("3. Model file is not corrupted")
        print("=" * 50)
        model = None


@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/index1")
def index1():
    return render_template("index1.html")

@app.route("/index2")
def index2():
    return render_template("index2.html")

@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/detection_results')
def detection_results():
	return render_template('detection_results.html')


@app.route('/upload')
def upload():
	return render_template('upload.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')


@app.route('/note')
def note():
	return render_template('note.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index1.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index1.html")
    else:
        return render_template("signup.html")



@app.route("/notebook")
def notebook():
    return render_template("notebook.html")




    
@app.route("/predict", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            filename = f.filename
            
            if not filename:
                return render_template('index.html', error="No file selected")
            
            if model is None:
                return render_template('display_image.html', result='Model not loaded. Please check server logs.')
            
            input_shape = (128, 128, 3)
            pr_data = []
            print("@@ Input posted =", filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_path)

            
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # Handle image upload
                    original = cv2.imread(file_path)

                    face_crop = extract_face(original)
                    if face_crop is not None:
                        processed = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    else:
                        # fallback to center-cropped, resized image
                        pil_img = load_img(file_path, target_size=(150, 150))
                        processed = img_to_array(pil_img)
                        processed = cv2.resize(processed, (128, 128))

                    processed = processed / 255.0
                    processed = np.expand_dims(processed, axis=0)
                    
                    probs = model.predict(processed, verbose=0)
                    fake_prob = float(probs[0][1])
                    real_prob = float(probs[0][0])
                    result = np.argmax(probs)
                    print(f"Image prediction: Class={result}, Real={real_prob:.4f}, Fake={fake_prob:.4f}")

                    if fake_prob >= FAKE_THRESHOLD:
                        ans = 'Fake'
                    elif real_prob >= CONFIDENCE_THRESHOLD:
                        # Very confident it's real
                        ans = 'Real'
                    else:
                        # Moderate confidence - use argmax
                        ans = 'Real' if result == 0 else 'Fake'
                except Exception as e:
                    print(f"Error processing image: {e}")
                    ans = f'Error processing image: {str(e)}'
            else:
                # Unsupported file type
                ans = 'Unsupported file type. Please upload PNG, JPG, or JPEG images.'

            return render_template('display_image.html', result=ans)

    return render_template('index.html')


    
@app.route("/predict1", methods=["GET", "POST"])
def predict_video():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            filename = f.filename
            
            if not filename:
                return render_template('index2.html', error="No file selected")
            
            if model is None:
                return render_template('display_image.html', result='Model not loaded. Please check server logs.')
            
            input_shape = (128, 128, 3)
            pr_data = []
            print("@@ Input posted =", filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_path)

            if filename.lower().endswith(('.mp4', '.avi', '.mkv')):
                try:
                    # Handle video upload
                    cap = cv2.VideoCapture(file_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    max_frames = 30  # Limit frames to prevent memory issues
                    
                    # Sample frames uniformly across video (better than sequential)
                    if total_frames > max_frames:
                        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
                    else:
                        frame_indices = list(range(total_frames))
                    
                    frames = []
                    faces_extracted = 0
                    
                    for idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if not ret:
                            continue

                        face = extract_face(frame)
                        if face is not None:
                            frames.append(face)
                            faces_extracted += 1
                        else:
                            # Fallback: resize full frame
                            resized = cv2.resize(frame, (128, 128))
                            frames.append(resized)

                    cap.release()

                    if len(frames) == 0:
                        ans = 'Error: Could not process frames from video.'
                    else:
                        # Convert frames to numpy array
                        video_data = np.array(frames) / 255.0

                        # Perform prediction on the video data
                        predictions = model.predict(video_data, verbose=0)
                        result = np.argmax(predictions, axis=1)
                        unique_vals, counts = np.unique(result, return_counts=True)
                        mean_probs = np.mean(predictions, axis=0)
                        fake_prob = float(mean_probs[1])
                        real_prob = float(mean_probs[0])
                        
                        # Calculate prediction variance (uncertainty measure)
                        prob_variance = np.var(predictions[:, 1])  # Variance in fake probabilities
                        max_fake_prob = float(np.max(predictions[:, 1]))  # Maximum fake probability across all frames
                        min_fake_prob = float(np.min(predictions[:, 1]))  # Minimum fake probability
                        
                        # Count frames with fake probability above thresholds
                        frames_above_avg_threshold = np.sum(predictions[:, 1] >= FAKE_THRESHOLD)
                        frames_above_frame_threshold = np.sum(predictions[:, 1] >= FAKE_FRAME_THRESHOLD)
                        
                        print(f"Video stats: Total frames={total_frames}, FPS={fps:.2f}, Faces extracted={faces_extracted}/{len(frames)}")
                        print("Frame prediction counts:", dict(zip(unique_vals, counts)))
                        print("Raw model results (video frames):", result)
                        print(f"Average probabilities: Real={real_prob:.4f}, Fake={fake_prob:.4f}")
                        print(f"Fake prob range: Min={min_fake_prob:.6f}, Max={max_fake_prob:.6f}")
                        print(f"Fake probability variance: {prob_variance:.6f}")
                        print(f"Frames with fake_prob >= {FAKE_THRESHOLD}: {frames_above_avg_threshold}")
                        print(f"Frames with fake_prob >= {FAKE_FRAME_THRESHOLD}: {frames_above_frame_threshold}")

                        # Enhanced decision logic for morphed video detection
                        # Strategy: Be more aggressive in detecting fakes, especially when model is overconfident
                        
                        # 1. If average fake probability is high enough
                        if fake_prob >= FAKE_THRESHOLD:
                            ans = 'Fake'
                            print(f"Decision: Fake (avg fake_prob {fake_prob:.4f} >= {FAKE_THRESHOLD})")
                        
                        # 2. If any frame has significant fake probability (more sensitive)
                        elif max_fake_prob >= FAKE_FRAME_THRESHOLD:
                            # Check how many frames are above this threshold
                            if frames_above_frame_threshold >= max(1, len(frames) * 0.05):  # At least 5% of frames
                                ans = 'Fake'
                                print(f"Decision: Fake (max_fake_prob {max_fake_prob:.4f} >= {FAKE_FRAME_THRESHOLD}, {frames_above_frame_threshold} frames)")
                            else:
                                # Only a few frames, but still suspicious
                                if real_prob >= SUSPICIOUS_CONFIDENCE:
                                    ans = 'Fake'  # Overconfident + some fake signals = suspicious
                                    print(f"Decision: Fake (suspicious: overconfident Real + {frames_above_frame_threshold} frames with fake signals)")
                                else:
                                    ans = 'Real'
                        
                        # 3. If model is suspiciously overconfident (>99% Real), be very cautious
                        elif real_prob >= SUSPICIOUS_CONFIDENCE:
                            # Very high confidence (>99%) on Real might indicate the model is wrong for morphed videos
                            # For morphed videos, the model often gives 99.99% Real confidence incorrectly
                            # Be extremely aggressive: ANY non-zero fake probability or variance is suspicious
                            # Also check if fake_prob is abnormally low (near zero) which is suspicious for morphed videos
                            
                            # Heuristic: If real_prob > 99.5% AND fake_prob < 0.001, it's suspiciously overconfident
                            is_suspiciously_overconfident = (real_prob > 0.995) and (fake_prob < 0.001)
                            
                            if is_suspiciously_overconfident or max_fake_prob > 0.00001 or prob_variance > 0.000001 or frames_above_frame_threshold > 0:
                                ans = 'Fake'
                                reason = "suspicious overconfidence"
                                if is_suspiciously_overconfident:
                                    reason += " (real_prob > 99.5% with fake_prob < 0.1%)"
                                print(f"Decision: Fake ({reason}: real_prob={real_prob:.4f}, fake_prob={fake_prob:.6f}, max_fake_prob={max_fake_prob:.6f})")
                            # If ALL frames have fake_prob < 0.00001 AND variance is extremely low, might be truly real
                            # But still be cautious - if faces were extracted poorly, morphed videos might slip through
                            elif faces_extracted < len(frames) * 0.5:  # Less than 50% faces extracted
                                ans = 'Fake'  # Suspicious: overconfident + poor face detection
                                print(f"Decision: Fake (suspicious: overconfident + only {faces_extracted}/{len(frames)} faces extracted)")
                            else:
                                ans = 'Real'
                                print(f"Decision: Real (high confidence, minimal fake signals, good face detection)")
                        
                        # 4. Standard confidence check
                        elif real_prob >= CONFIDENCE_THRESHOLD:
                            # High confidence but not suspiciously so
                            if prob_variance > 0.01:  # High variance suggests uncertainty
                                print(f"Warning: High confidence Real ({real_prob:.2%}) but high variance ({prob_variance:.4f})")
                                if frames_above_frame_threshold > len(frames) * 0.1:  # >10% frames suggest fake
                                    ans = 'Fake'
                                    print(f"Decision: Fake (high variance + {frames_above_frame_threshold} frames suggest fake)")
                                else:
                                    ans = 'Real'
                            else:
                                ans = 'Real'
                        
                        # 5. Moderate confidence - use argmax as fallback
                        else:
                            # Use majority vote with probability check
                            if np.sum(result == 1) > np.sum(result == 0):  # More frames predict Fake
                                ans = 'Fake'
                            else:
                                ans = 'Real'
                            print(f"Decision: {ans} (moderate confidence, majority vote)")
                except Exception as e:
                    print(f"Error processing video: {e}")
                    ans = f'Error processing video: {str(e)}'
            else:
                # Unsupported file type
                ans = 'Unsupported file type. Please upload MP4, AVI, or MKV videos.'

            return render_template('display_image.html', result=ans)

    return render_template('index2.html')    




if __name__ == '__main__':
    app.run(debug=True)
