from flask import Flask, render_template, Response, jsonify, request
import cv2
import sys
import mediapipe as mp
import numpy as np
import time
import json
import threading
from ultralytics import YOLO
import base64
from datetime import datetime

app = Flask(__name__)

# Global variables for detection state
detection_state = {
    'is_running': False,
    'face_count': 0,
    'head_pose': 'No face detected',
    'eye_status': 'No face detected',
    'gaze_direction': 'No face detected',
    'detected_objects': [],
    'alerts': [],
    'activity_log': []
}

# Separate variable for frame storage (not included in JSON responses)
current_frame = None
frame_ready = False

# Initialize detection models
def initialize_models():
    """Initialize all detection models"""
    try:
        # MediaPipe models
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        
        face_detection = mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=0.5
        )
        
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # YOLOv5 model
        yolo_model = YOLO('yolov5s.pt')
        
        return face_detection, face_mesh, yolo_model
    except Exception as e:
        print(f"Error initializing models: {e}")
        return None, None, None

# Detection functions
def calculate_ear(eye_points):
    """Calculate the Eye Aspect Ratio (EAR) for blinking detection"""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_eye_gaze(eye_points, frame_width, frame_height):
    """Calculate eye gaze direction based on eye landmarks"""
    eye_center_x = np.mean(eye_points[:, 0])
    eye_center_y = np.mean(eye_points[:, 1])
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    gaze_x = (eye_center_x - frame_center_x) / frame_center_x
    gaze_y = (eye_center_y - frame_center_y) / frame_center_y
    return gaze_x, gaze_y

def add_alert(alert_type, message, priority='medium'):
    """Add an alert to the detection state"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    alert = {
        'type': alert_type,
        'message': message,
        'priority': priority,
        'timestamp': timestamp
    }
    detection_state['alerts'].insert(0, alert)
    
    # Keep only last 10 alerts
    if len(detection_state['alerts']) > 10:
        detection_state['alerts'] = detection_state['alerts'][:10]

def add_activity_log(event, status='info'):
    """Add an activity to the log"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = {
        'time': timestamp,
        'event': event,
        'status': status
    }
    detection_state['activity_log'].insert(0, log_entry)
    
    # Keep only last 50 log entries
    if len(detection_state['activity_log']) > 50:
        detection_state['activity_log'] = detection_state['activity_log'][:50]

def detection_thread():
    """Main detection thread"""
    global detection_state, current_frame, frame_ready
    
    # Initialize models
    face_detection, face_mesh, yolo_model = initialize_models()
    if not all([face_detection, face_mesh, yolo_model]):
        print("Failed to initialize models")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Detection variables
    EAR_THRESHOLD = 0.21
    frame_count = 0
    eyes_off_screen_start = None
    object_alert_start = None
    OBJECT_DETECTION_INTERVAL = 10
    EYES_OFF_SCREEN_THRESHOLD = 5.0
    OBJECT_ALERT_THRESHOLD = 3.0
    
    # Objects of interest
    objects_of_interest = [
        'cell phone', 'phone', 'mobile phone', 'smartphone',
        'book', 'notebook', 'laptop', 'computer',
        'remote', 'controller', 'tablet', 'ipad'
    ]
    
    # 3D model points for head pose estimation
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)
    
    print("Detection thread started")
    
    while detection_state['is_running']:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Object detection
        if frame_count % OBJECT_DETECTION_INTERVAL == 0:
            try:
                results = yolo_model(frame, verbose=False)
                detected_objects = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = yolo_model.names[class_id]
                            
                            if any(obj in class_name.lower() for obj in objects_of_interest):
                                detected_objects.append({
                                    'name': class_name,
                                    'confidence': float(confidence),
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                                })
                
                detection_state['detected_objects'] = detected_objects
                
                # Handle object alerts
                if detected_objects:
                    if object_alert_start is None:
                        object_alert_start = current_time
                    else:
                        time_with_objects = current_time - object_alert_start
                        if time_with_objects > OBJECT_ALERT_THRESHOLD:
                            object_names = [obj['name'] for obj in detected_objects]
                            add_alert('Suspicious Objects', f"Detected for {time_with_objects:.1f}s: {', '.join(object_names)}", 'high')
                else:
                    object_alert_start = None
                    
            except Exception as e:
                print(f"Error in object detection: {e}")
        
        # Face detection and analysis
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = face_detection.process(rgb_frame)
        mesh_results = face_mesh.process(rgb_frame)
        
        # Update face count
        face_count = 0
        if detection_results.detections:
            face_count = len(detection_results.detections)
            detection_state['face_count'] = face_count
            
            # Multiple faces alert
            if face_count > 1:
                add_alert('Multiple Faces', f"{face_count} faces detected - potential collaboration", 'high')
                add_activity_log(f"Multiple faces detected ({face_count})", 'high')
            elif face_count == 1:
                add_activity_log("Single face detected", 'normal')
        
        # Eye tracking and head pose for single face
        if mesh_results.multi_face_landmarks and len(mesh_results.multi_face_landmarks) == 1:
            landmarks = mesh_results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # Eye landmarks
            left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            
            left_eye_points = np.array([(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) 
                                      for i in left_eye_indices])
            right_eye_points = np.array([(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) 
                                       for i in right_eye_indices])
            
            # Eye status - only show if eyes are open
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0
            
            if avg_ear >= EAR_THRESHOLD:
                detection_state['eye_status'] = "Eyes Open"
            else:
                detection_state['eye_status'] = "Eyes Closed"
            
            # Gaze detection
            left_gaze_x, left_gaze_y = calculate_eye_gaze(left_eye_points, w, h)
            right_gaze_x, right_gaze_y = calculate_eye_gaze(right_eye_points, w, h)
            avg_gaze_x = (left_gaze_x + right_gaze_x) / 2.0
            avg_gaze_y = (left_gaze_y + right_gaze_y) / 2.0
            
            gaze_threshold = 0.3
            if abs(avg_gaze_x) < gaze_threshold and abs(avg_gaze_y) < gaze_threshold:
                detection_state['gaze_direction'] = "Looking at Screen"
                if eyes_off_screen_start is not None:
                    eyes_off_screen_start = None
            else:
                detection_state['gaze_direction'] = "Looking Away"
                if eyes_off_screen_start is None:
                    eyes_off_screen_start = current_time
            
            # Eyes off screen alert
            if eyes_off_screen_start is not None:
                time_off_screen = current_time - eyes_off_screen_start
                if time_off_screen > EYES_OFF_SCREEN_THRESHOLD:
                    add_alert('Eyes Off Screen', f"Looking away for {time_off_screen:.1f} seconds", 'medium')
            
            # Head pose estimation
            image_points = np.array([
                (int(landmarks.landmark[1].x * w), int(landmarks.landmark[1].y * h)),
                (int(landmarks.landmark[152].x * w), int(landmarks.landmark[152].y * h)),
                (int(landmarks.landmark[226].x * w), int(landmarks.landmark[226].y * h)),
                (int(landmarks.landmark[446].x * w), int(landmarks.landmark[446].y * h)),
                (int(landmarks.landmark[57].x * w), int(landmarks.landmark[57].y * h)),
                (int(landmarks.landmark[287].x * w), int(landmarks.landmark[287].y * h))
            ], dtype=np.float64)
            
            focal_length = w
            camera_matrix = np.array([
                [focal_length, 0, w/2],
                [0, focal_length, h/2],
                [0, 0, 1]
            ], dtype=np.float64)
            
            dist_coeffs = np.zeros((4, 1))
            
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )
            
            if success:
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                pitch = np.degrees(np.arctan2(-rotation_mat[2, 1], rotation_mat[2, 2]))
                yaw = np.degrees(np.arctan2(rotation_mat[2, 0], np.sqrt(rotation_mat[2, 1]**2 + rotation_mat[2, 2]**2)))
                roll = np.degrees(np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]))
                
                # Determine head pose direction (reversed left/right)
                if yaw < -15:
                    detection_state['head_pose'] = "Looking Right"
                elif yaw > 15:
                    detection_state['head_pose'] = "Looking Left"
                elif pitch < -10:
                    detection_state['head_pose'] = "Looking Up"
                elif pitch > 10:
                    detection_state['head_pose'] = "Looking Down"
                else:
                    detection_state['head_pose'] = "Looking Forward"
        
        # Draw detection results on frame
        if detection_results.detections:
            for i, detection in enumerate(detection_results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                color = (0, 0, 255) if face_count > 1 else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                confidence = detection.score[0]
                cv2.putText(frame, f'Face {i+1}: {confidence:.2f}', 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, color, 2)
        
        # Draw detected objects
        for obj in detection_state['detected_objects']:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{obj['name']}: {obj['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add status text to frame
        cv2.putText(frame, f'Faces: {detection_state["face_count"]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Head: {detection_state["head_pose"]}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'Eyes: {detection_state["eye_status"]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Gaze: {detection_state["gaze_direction"]}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Store frame for web display (using separate variables)
        current_frame = frame.copy()
        frame_ready = True
        
        time.sleep(0.033)  # ~30 FPS
    
    # Cleanup
    cap.release()
    face_detection.close()
    face_mesh.close()
    print("Detection thread stopped")

def generate_frames():
    """Generate video frames for streaming"""
    global current_frame, frame_ready
    while True:
        if frame_ready and current_frame is not None:
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Send a placeholder frame when no webcam is active
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, 'Webcam not active', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

# Flask routes
@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get current detection status"""
    return jsonify(detection_state)

@app.route('/api/start')
def start_detection():
    """Start the detection system"""
    global detection_state
    if not detection_state['is_running']:
        detection_state['is_running'] = True
        detection_state['alerts'] = []
        detection_state['activity_log'] = []
        add_activity_log("Monitoring started", 'normal')
        
        # Start detection thread
        thread = threading.Thread(target=detection_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'success', 'message': 'Detection started'})
    return jsonify({'status': 'error', 'message': 'Detection already running'})

@app.route('/api/stop')
def stop_detection():
    """Stop the detection system"""
    global detection_state, frame_ready
    detection_state['is_running'] = False
    frame_ready = False
    add_activity_log("Monitoring stopped", 'normal')
    return jsonify({'status': 'success', 'message': 'Detection stopped'})

@app.route('/api/clear-alerts')
def clear_alerts():
    """Clear all alerts"""
    detection_state['alerts'] = []
    return jsonify({'status': 'success', 'message': 'Alerts cleared'})

@app.route('/api/clear-log')
def clear_log():
    """Clear activity log"""
    detection_state['activity_log'] = []
    return jsonify({'status': 'success', 'message': 'Log cleared'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 