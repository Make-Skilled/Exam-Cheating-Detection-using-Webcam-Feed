import cv2
import sys
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO

def calculate_ear(eye_points):
    """
    Calculate the Eye Aspect Ratio (EAR) for blinking detection
    """
    # Vertical distances
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    # Horizontal distance
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_eye_gaze(eye_points, frame_width, frame_height):
    """
    Calculate eye gaze direction based on eye landmarks
    """
    # Calculate eye center
    eye_center_x = np.mean(eye_points[:, 0])
    eye_center_y = np.mean(eye_points[:, 1])
    
    # Calculate gaze direction relative to frame center
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    # Normalize gaze direction
    gaze_x = (eye_center_x - frame_center_x) / frame_center_x
    gaze_y = (eye_center_y - frame_center_y) / frame_center_y
    
    return gaze_x, gaze_y

def main():
    """
    Main function to capture and display webcam feed with face detection, head pose estimation, eye tracking, and object detection
    """
    # Initialize MediaPipe Face Detection and Face Mesh
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize YOLOv5 model for object detection
    try:
        model = YOLO('yolov5s.pt')  # Load pre-trained YOLOv5 small model
        print("✅ YOLOv5 model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading YOLOv5 model: {e}")
        print("Please ensure you have internet connection for model download")
        sys.exit(1)
    
    # Define objects of interest for exam cheating detection
    objects_of_interest = [
        'cell phone', 'phone', 'mobile phone', 'smartphone',
        'book', 'notebook', 'laptop', 'computer',
        'remote', 'controller', 'tablet', 'ipad'
    ]
    
    # Initialize the face detection model
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0, 
        min_detection_confidence=0.5
    )
    
    # Initialize the face mesh model for landmarks
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 3D model points for head pose estimation
    # These are the 3D coordinates of facial landmarks in a standard face model
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)
    
    # Eye tracking variables
    EAR_THRESHOLD = 0.21  # Eye Aspect Ratio threshold for blinking
    BLINK_CONSEC_FRAMES = 2  # Number of consecutive frames for blink detection
    EYES_OFF_SCREEN_THRESHOLD = 5.0  # Seconds threshold for eyes off screen alert
    
    # Object detection variables
    OBJECT_DETECTION_INTERVAL = 10  # Process object detection every N frames
    frame_count = 0
    detected_objects = []
    object_alert_start = None
    OBJECT_ALERT_THRESHOLD = 3.0  # Seconds threshold for object detection alert
    
    # Tracking variables
    left_ear_history = []
    right_ear_history = []
    eyes_off_screen_start = None
    last_blink_time = time.time()
    blink_count = 0
    
    # Initialize the webcam (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)
    
    print("Webcam feed with comprehensive detection and tracking started. Press 'q' to quit.")
    print("⚠️  WARNING: Multiple faces detected!" if False else "✅ Single face detected or no faces detected.")
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # If frame is read correctly ret is True
            if not ret:
                print("Error: Can't receive frame from webcam")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Object detection (process every N frames for performance)
            if frame_count % OBJECT_DETECTION_INTERVAL == 0:
                try:
                    # Run YOLOv5 inference
                    results = model(frame, verbose=False)
                    
                    # Process results
                    detected_objects = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                # Get box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Get confidence and class
                                confidence = float(box.conf[0])
                                class_id = int(box.cls[0])
                                class_name = model.names[class_id]
                                
                                # Check if object is of interest
                                if any(obj in class_name.lower() for obj in objects_of_interest):
                                    detected_objects.append({
                                        'name': class_name,
                                        'confidence': confidence,
                                        'bbox': (x1, y1, x2, y2)
                                    })
                    
                    # Handle object detection alerts
                    if detected_objects:
                        if object_alert_start is None:
                            object_alert_start = current_time
                        else:
                            time_with_objects = current_time - object_alert_start
                            if time_with_objects > OBJECT_ALERT_THRESHOLD:
                                object_names = [obj['name'] for obj in detected_objects]
                                print(f"🚨 ALERT: Suspicious objects detected for {time_with_objects:.1f} seconds: {', '.join(object_names)}")
                    else:
                        object_alert_start = None
                        
                except Exception as e:
                    print(f"Error in object detection: {e}")
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect faces
            detection_results = face_detection.process(rgb_frame)
            mesh_results = face_mesh.process(rgb_frame)
            
            # Count faces and handle warnings
            face_count = 0
            if detection_results.detections:
                face_count = len(detection_results.detections)
                
                # Print warning if more than one face is detected
                if face_count > 1:
                    print(f"⚠️  WARNING: Multiple faces detected! Count: {face_count}")
                elif face_count == 1:
                    print(f"✅ Single face detected. Count: {face_count}")
                else:
                    print(f"ℹ️  No faces detected. Count: {face_count}")
            
            # Eye tracking and head pose estimation for single face
            head_pose_text = "No face detected"
            eye_status = "No face detected"
            gaze_direction = "No face detected"
            
            if mesh_results.multi_face_landmarks and len(mesh_results.multi_face_landmarks) == 1:
                landmarks = mesh_results.multi_face_landmarks[0]
                
                # Get image dimensions
                h, w, _ = frame.shape
                
                # Extract eye landmarks for blinking detection
                # Left eye landmarks (MediaPipe face mesh indices)
                left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                # Right eye landmarks
                right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                
                # Extract eye points
                left_eye_points = np.array([(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) 
                                          for i in left_eye_indices])
                right_eye_points = np.array([(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) 
                                           for i in right_eye_indices])
                
                # Calculate EAR for both eyes
                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)
                
                # Average EAR
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Detect blinking
                if avg_ear < EAR_THRESHOLD:
                    blink_count += 1
                    last_blink_time = current_time
                    eye_status = "Blinking"
                else:
                    eye_status = "Eyes Open"
                
                # Calculate eye gaze direction
                left_gaze_x, left_gaze_y = calculate_eye_gaze(left_eye_points, w, h)
                right_gaze_x, right_gaze_y = calculate_eye_gaze(right_eye_points, w, h)
                
                # Average gaze direction
                avg_gaze_x = (left_gaze_x + right_gaze_x) / 2.0
                avg_gaze_y = (left_gaze_y + right_gaze_y) / 2.0
                
                # Determine gaze direction
                gaze_threshold = 0.3
                if abs(avg_gaze_x) < gaze_threshold and abs(avg_gaze_y) < gaze_threshold:
                    gaze_direction = "Looking at Screen"
                    # Reset eyes off screen timer
                    if eyes_off_screen_start is not None:
                        eyes_off_screen_start = None
                else:
                    gaze_direction = "Looking Away"
                    # Start timer if eyes are off screen
                    if eyes_off_screen_start is None:
                        eyes_off_screen_start = current_time
                
                # Check for eyes off screen alert
                if eyes_off_screen_start is not None:
                    time_off_screen = current_time - eyes_off_screen_start
                    if time_off_screen > EYES_OFF_SCREEN_THRESHOLD:
                        print(f"🚨 ALERT: Eyes off screen for {time_off_screen:.1f} seconds!")
                
                # Extract 2D points for head pose estimation
                # MediaPipe face mesh landmark indices for key facial points
                image_points = np.array([
                    (int(landmarks.landmark[1].x * w), int(landmarks.landmark[1].y * h)),      # Nose tip
                    (int(landmarks.landmark[152].x * w), int(landmarks.landmark[152].y * h)),  # Chin
                    (int(landmarks.landmark[226].x * w), int(landmarks.landmark[226].y * h)),  # Left eye left corner
                    (int(landmarks.landmark[446].x * w), int(landmarks.landmark[446].y * h)),  # Right eye right corner
                    (int(landmarks.landmark[57].x * w), int(landmarks.landmark[57].y * h)),    # Left mouth corner
                    (int(landmarks.landmark[287].x * w), int(landmarks.landmark[287].y * h))   # Right mouth corner
                ], dtype=np.float64)
                
                # Camera matrix (assuming a standard webcam)
                focal_length = w
                camera_matrix = np.array([
                    [focal_length, 0, w/2],
                    [0, focal_length, h/2],
                    [0, 0, 1]
                ], dtype=np.float64)
                
                # Distortion coefficients (assuming no distortion)
                dist_coeffs = np.zeros((4, 1))
                
                # Solve PnP to get rotation and translation vectors
                success, rotation_vec, translation_vec = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs
                )
                
                if success:
                    # Convert rotation vector to rotation matrix
                    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                    
                    # Get Euler angles (pitch, yaw, roll)
                    # Pitch: up/down movement
                    # Yaw: left/right movement
                    # Roll: head tilt
                    pitch = np.degrees(np.arctan2(-rotation_mat[2, 1], rotation_mat[2, 2]))
                    yaw = np.degrees(np.arctan2(rotation_mat[2, 0], np.sqrt(rotation_mat[2, 1]**2 + rotation_mat[2, 2]**2)))
                    roll = np.degrees(np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]))
                    
                    # Determine head pose direction
                    head_pose_text = "Looking Forward"
                    
                    # Yaw (left/right)
                    if yaw < -15:
                        head_pose_text = "Looking Left"
                    elif yaw > 15:
                        head_pose_text = "Looking Right"
                    # Pitch (up/down)
                    elif pitch < -10:
                        head_pose_text = "Looking Up"
                    elif pitch > 10:
                        head_pose_text = "Looking Down"
                    
                    # Print head pose and eye tracking information
                    print(f"Head Pose: {head_pose_text} (Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°, Roll: {roll:.1f}°) | Eye Status: {eye_status} | Gaze: {gaze_direction} | EAR: {avg_ear:.3f}")
                
                # Draw eye landmarks
                for point in left_eye_points:
                    cv2.circle(frame, tuple(point), 1, (0, 255, 0), -1)
                for point in right_eye_points:
                    cv2.circle(frame, tuple(point), 1, (0, 255, 0), -1)
            
            # Draw detected objects
            for obj in detected_objects:
                x1, y1, x2, y2 = obj['bbox']
                confidence = obj['confidence']
                name = obj['name']
                
                # Draw bounding box (red for suspicious objects)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw label
                label = f"{name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw face detection results
            if detection_results.detections:
                for i, detection in enumerate(detection_results.detections):
                    # Get bounding box coordinates
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    
                    # Convert relative coordinates to absolute coordinates
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    
                    # Choose color based on face count (red for multiple faces, green for single)
                    color = (0, 0, 255) if face_count > 1 else (0, 255, 0)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw confidence score and face number
                    confidence = detection.score[0]
                    cv2.putText(frame, f'Face {i+1}: {confidence:.2f}', 
                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, color, 2)
                    
                    # Draw key points (eyes, nose, mouth)
                    for keypoint in detection.location_data.relative_keypoints:
                        kp_x = int(keypoint.x * iw)
                        kp_y = int(keypoint.y * ih)
                        cv2.circle(frame, (kp_x, kp_y), 2, (255, 0, 0), -1)
            
            # Display information on frame
            cv2.putText(frame, f'Faces Detected: {face_count}', 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, (255, 255, 255), 2)
            
            # Display head pose information
            cv2.putText(frame, f'Head Pose: {head_pose_text}', 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, (255, 255, 0), 2)
            
            # Display eye status
            eye_color = (0, 255, 0) if eye_status == "Eyes Open" else (0, 0, 255)
            cv2.putText(frame, f'Eye Status: {eye_status}', 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, eye_color, 2)
            
            # Display gaze direction
            gaze_color = (0, 255, 0) if gaze_direction == "Looking at Screen" else (0, 0, 255)
            cv2.putText(frame, f'Gaze: {gaze_direction}', 
                      (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, gaze_color, 2)
            
            # Display blink count
            cv2.putText(frame, f'Blinks: {blink_count}', 
                      (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, (255, 255, 255), 2)
            
            # Display eyes off screen timer
            if eyes_off_screen_start is not None:
                time_off_screen = current_time - eyes_off_screen_start
                cv2.putText(frame, f'Eyes Off Screen: {time_off_screen:.1f}s', 
                          (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 0, 255), 2)
            
            # Display object detection information
            if detected_objects:
                object_names = [obj['name'] for obj in detected_objects]
                cv2.putText(frame, f'Objects: {", ".join(object_names)}', 
                          (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 0, 255), 2)
                
                # Display object alert timer
                if object_alert_start is not None:
                    time_with_objects = current_time - object_alert_start
                    cv2.putText(frame, f'Objects Detected: {time_with_objects:.1f}s', 
                              (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 0, 255), 2)
            
            # Display warning text if multiple faces
            if face_count > 1:
                cv2.putText(frame, 'MULTIPLE FACES DETECTED!', 
                          (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Webcam Feed with Comprehensive Detection and Tracking', frame)
            
            # Wait for key press and check if 'q' is pressed
            # cv2.waitKey(1) returns the ASCII value of the key pressed
            # & 0xFF is used to get the last 8 bits of the number
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Release everything when job is finished
        cap.release()
        cv2.destroyAllWindows()
        face_detection.close()
        face_mesh.close()
        print("Webcam feed stopped.")

if __name__ == "__main__":
    main() 