# Exam Cheating Detection using Webcam Feed

A Python project that uses OpenCV, MediaPipe, and YOLOv5 to access the webcam, display live video feed, detect faces with bounding boxes, monitor for multiple faces, estimate head pose orientation, track eye movements with blinking detection, and detect suspicious objects. This serves as the foundation for comprehensive exam cheating detection systems.

## Features

- Real-time webcam feed capture
- Live video display using OpenCV
- Face detection using MediaPipe
- **Head pose estimation** using facial landmarks
- **Head orientation detection** (Looking Left, Right, Up, Down, Forward)
- **Eye tracking and blinking detection** using EAR (Eye Aspect Ratio)
- **Eye gaze direction detection** (Looking at Screen vs Looking Away)
- **Eyes off screen alert system** (5-second threshold)
- **Blink counting and monitoring**
- **YOLOv5 object detection** for suspicious objects
- **Suspicious object alert system** (3-second threshold)
- **Real-time object tracking** with bounding boxes
- **Flask web dashboard** with real-time monitoring
- **RESTful API** for system control and status
- Face counting and monitoring
- **Multiple face detection warnings** (console alerts)
- Dynamic bounding box colors (green for single face, red for multiple faces)
- Bounding boxes around detected faces
- Confidence scores for each detection
- Facial keypoint visualization (eyes, nose, mouth)
- **Eye landmark visualization** (green dots on eye contours)
- **Object bounding box visualization** (red boxes for suspicious objects)
- Real-time face count display on video feed
- **Real-time head pose angles** (Pitch, Yaw, Roll)
- **Real-time eye status and gaze information**
- **Real-time object detection information**
- Clean exit with 'q' key or Ctrl+C
- Error handling for webcam access issues

## Requirements

- Python 3.7 or higher
- OpenCV (opencv-python)
- MediaPipe
- NumPy
- PyTorch
- TorchVision
- Ultralytics (YOLOv5)
- Flask
- Webcam access

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Exam-Cheating-Detection-using-Webcam-Feed
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The first time you run the application, it will automatically download the YOLOv5 model (~14MB) from the internet.

## Usage

### Option 1: Command Line Interface (CLI)
Run the main script to start the webcam feed with comprehensive detection and tracking:

```bash
python main.py
```

### Option 2: Web Dashboard (Recommended)
Run the Flask web application for a modern dashboard interface:

```bash
python run_flask.py
```

Then open your browser and navigate to: `http://localhost:5000`

### Controls

#### CLI Controls:
- **'q' key**: Quit the application
- **Ctrl+C**: Force quit (alternative method)

#### Web Dashboard Controls:
- **Start Button**: Begin monitoring and detection
- **Stop Button**: Stop monitoring and detection
- **Clear Alerts**: Remove all active alerts
- **Clear Log**: Clear activity log history

## Web Dashboard Features

### **Real-time Monitoring Interface**
- **Live webcam feed** with detection overlays
- **Real-time status updates** every second
- **Interactive controls** for system management
- **Responsive design** for all screen sizes

### **Detection Status Panel**
- **Face count** with real-time updates
- **Head pose** direction display
- **Eye status** (open/blinking) monitoring
- **Blink counter** with live tracking

### **Alert Management**
- **Color-coded alerts** by priority level
- **Real-time alert updates** as events occur
- **Alert clearing** functionality
- **Timestamp tracking** for all alerts

### **Activity Logging**
- **Comprehensive event logging** with timestamps
- **Status indicators** for each event type
- **Scrollable log** with sticky headers
- **Log clearing** functionality

### **API Endpoints**
- `GET /api/status` - Get current detection status
- `POST /api/start` - Start detection system
- `POST /api/stop` - Stop detection system
- `POST /api/clear-alerts` - Clear all alerts
- `POST /api/clear-log` - Clear activity log

## Face Detection Features

The application will:
- Draw **green bounding boxes** around detected faces when only one face is present
- Draw **red bounding boxes** around detected faces when multiple faces are detected
- Display confidence scores for each face detection
- Show blue dots for facial keypoints (eyes, nose, mouth)
- Process frames in real-time for smooth detection

## Face Counting and Monitoring

The application provides:
- **Real-time face counting** displayed on the video feed
- **Console warnings** when multiple faces are detected:
  - `⚠️  WARNING: Multiple faces detected! Count: X`
  - `✅ Single face detected. Count: 1`
  - `ℹ️  No faces detected. Count: 0`
- **Visual warnings** on the video feed when multiple faces are detected
- **Numbered face labels** (Face 1, Face 2, etc.) for easy identification

## Head Pose Estimation

The application provides advanced head pose analysis:
- **Real-time head orientation detection**:
  - "Looking Forward" (normal exam position)
  - "Looking Left" (potential distraction)
  - "Looking Right" (potential distraction)
  - "Looking Up" (potential distraction)
  - "Looking Down" (potential distraction)
- **Precise angle measurements**:
  - **Pitch**: Up/down head movement (degrees)
  - **Yaw**: Left/right head movement (degrees)
  - **Roll**: Head tilt (degrees)
- **Console output** with detailed angle information
- **Visual display** of current head pose on video feed

## Eye Tracking and Blinking Detection

The application provides comprehensive eye monitoring:
- **Eye Aspect Ratio (EAR) calculation** for precise blinking detection
- **Real-time blink detection** with configurable threshold (default: 0.21)
- **Blink counting** and display on video feed
- **Eye status monitoring**: "Eyes Open" vs "Blinking"
- **Eye landmark visualization** with green dots on eye contours

## Eye Gaze Direction Detection

The application tracks where the user is looking:
- **Gaze direction analysis** based on eye center position
- **Screen focus detection**: "Looking at Screen" vs "Looking Away"
- **Normalized gaze coordinates** for precise tracking
- **Configurable gaze threshold** (default: 0.3)

## Eyes Off Screen Alert System

The application includes an advanced alert system:
- **5-second threshold** for eyes off screen detection
- **Real-time timer display** showing time off screen
- **Console alerts** when threshold is exceeded:
  - `🚨 ALERT: Eyes off screen for X.X seconds!`
- **Visual countdown** on video feed
- **Automatic reset** when eyes return to screen

## YOLOv5 Object Detection

The application provides comprehensive object monitoring:
- **Real-time object detection** using YOLOv5 pre-trained model
- **Suspicious object detection** including:
  - Cell phones, smartphones, mobile phones
  - Books, notebooks
  - Laptops, computers
  - Tablets, iPads
  - Remotes, controllers
- **Object bounding boxes** with red color for suspicious objects
- **Confidence scores** for each detected object
- **Performance optimization** (processes every 10 frames)

## Suspicious Object Alert System

The application includes an advanced object monitoring system:
- **3-second threshold** for suspicious object detection
- **Real-time timer display** showing time with objects detected
- **Console alerts** when threshold is exceeded:
  - `🚨 ALERT: Suspicious objects detected for X.X seconds: object1, object2`
- **Visual object tracking** on video feed
- **Automatic reset** when objects are no longer detected

## Project Structure

```
Exam-Cheating-Detection-using-Webcam-Feed/
├── main.py              # Main CLI application with comprehensive detection and tracking
├── app.py               # Flask web application with API endpoints
├── run_flask.py         # Flask app runner script
├── requirements.txt     # Python dependencies
├── templates/
│   └── dashboard.html   # Web dashboard template
├── README.md           # Project documentation
└── yolov5su.pt         # YOLOv5 model file (auto-downloaded)
```

## Troubleshooting

### Webcam Access Issues

If you encounter webcam access problems:

1. **Permission Denied**: Ensure your application has permission to access the webcam
2. **Webcam in Use**: Close other applications that might be using the webcam
3. **Driver Issues**: Update your webcam drivers
4. **Multiple Webcams**: If you have multiple cameras, try changing the index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or higher

### OpenCV Installation Issues

If you have trouble installing OpenCV:

```bash
# Alternative installation methods
pip install opencv-python-headless  # Headless version (no GUI)
# or
conda install opencv  # If using Anaconda
```

### MediaPipe Installation Issues

If you encounter MediaPipe installation problems:

```bash
# Alternative installation methods
pip install mediapipe-silicon  # For Apple Silicon Macs
# or
conda install -c conda-forge mediapipe  # If using Anaconda
```

### PyTorch Installation Issues

If you encounter PyTorch installation problems:

```bash
# Alternative installation methods
conda install pytorch torchvision -c pytorch  # If using Anaconda
# or visit https://pytorch.org/ for platform-specific installation
```

### YOLOv5 Model Download Issues

If the YOLOv5 model fails to download:

1. **Check internet connection**
2. **Manual download**: The model will be downloaded to `~/.cache/ultralytics/`
3. **Alternative models**: You can specify different YOLOv5 models:
   - `yolov5n.pt` (nano - fastest, smallest)
   - `yolov5s.pt` (small - default)
   - `yolov5m.pt` (medium)
   - `yolov5l.pt` (large)
   - `yolov5x.pt` (extra large - most accurate)

### Flask Web App Issues

If you encounter Flask application problems:

1. **Port already in use**: Change the port in `run_flask.py` or `app.py`
2. **Template not found**: Ensure `dashboard.html` is in the `templates/` directory
3. **Module not found**: Install Flask with `pip install flask`

## Performance Tips

- **Detection Confidence**: Adjust `min_detection_confidence` in the code (default: 0.5) for different sensitivity levels
- **Model Selection**: Use `model_selection=0` for short-range detection or `model_selection=1` for full-range detection
- **Frame Processing**: The application processes every frame for real-time detection
- **Warning Frequency**: Console warnings are printed for each frame with multiple faces detected
- **Head Pose Sensitivity**: Adjust the angle thresholds in the code for different sensitivity levels:
  - Yaw threshold: ±15° (left/right detection)
  - Pitch threshold: ±10° (up/down detection)
- **Eye Tracking Sensitivity**: Adjust these parameters for different sensitivity levels:
  - EAR threshold: 0.21 (blinking detection sensitivity)
  - Gaze threshold: 0.3 (screen focus detection sensitivity)
  - Eyes off screen threshold: 5.0 seconds (alert timing)
- **Object Detection Performance**: Adjust these parameters for different performance levels:
  - Object detection interval: 10 frames (process every 10th frame)
  - Object alert threshold: 3.0 seconds (alert timing)
  - Objects of interest: Customize the list for your specific needs
- **Web Dashboard Performance**: 
  - Status updates every 1 second (configurable)
  - Automatic cleanup of old alerts and logs
  - Responsive design for optimal performance

## Exam Cheating Detection Logic

The system implements comprehensive cheating detection by:
1. **Monitoring face count** in real-time
2. **Alerting when multiple faces** are detected (potential collaboration)
3. **Tracking head orientation** to detect distractions or looking away from screen
4. **Monitoring eye movements** to detect looking away from screen
5. **Detecting blinking patterns** for attention monitoring
6. **Detecting suspicious objects** like phones, books, laptops
7. **Providing visual feedback** with color-coded bounding boxes
8. **Logging warnings** to console for monitoring purposes
9. **Measuring precise angles** for detailed analysis
10. **Timing eyes off screen** for extended distraction detection
11. **Timing suspicious object presence** for extended cheating detection
12. **Web-based monitoring** with real-time dashboard updates

### Detection Thresholds

#### Head Pose Detection:
- **Looking Left**: Yaw < -15°
- **Looking Right**: Yaw > 15°
- **Looking Up**: Pitch < -10°
- **Looking Down**: Pitch > 10°
- **Looking Forward**: Within normal ranges

#### Eye Tracking Detection:
- **Blinking**: EAR < 0.21
- **Looking at Screen**: Gaze deviation < 0.3
- **Eyes Off Screen Alert**: > 5.0 seconds

#### Object Detection:
- **Suspicious Objects**: Cell phones, books, laptops, tablets, etc.
- **Object Alert Threshold**: > 3.0 seconds
- **Detection Interval**: Every 10 frames (for performance)

## Console Output Examples

```
✅ YOLOv5 model loaded successfully
Head Pose: Looking Forward (Pitch: 2.3°, Yaw: -1.2°, Roll: 0.8°) | Eye Status: Eyes Open | Gaze: Looking at Screen | EAR: 0.245
🚨 ALERT: Eyes off screen for 5.2 seconds!
🚨 ALERT: Suspicious objects detected for 3.5 seconds: cell phone, book
```

## Web Dashboard Screenshots

The web dashboard provides:
- **Real-time webcam feed** with detection overlays
- **Live status panels** showing face count, head pose, eye status, and blink count
- **Active alerts panel** with color-coded priority levels
- **Activity log table** with timestamped events and status indicators
- **Interactive controls** for starting/stopping detection and clearing data

## Future Enhancements

This project can be extended with:
- Face recognition and identification
- **Advanced eye tracking** with pupil detection
- **Gaze point estimation** (exact screen coordinates)
- Multiple person detection and tracking
- Recording capabilities with detection overlays
- Real-time analysis and alerts
- Emotion detection
- **Advanced head pose tracking** with historical data
- **Machine learning models** for improved accuracy
- **Fatigue detection** based on blink patterns
- **Attention scoring** algorithms
- **Custom object detection models** for specific exam environments
- **Object tracking** across frames for better detection
- **Audio detection** for suspicious sounds
- **WebSocket support** for real-time dashboard updates
- **Database integration** for persistent logging
- **User authentication** for multi-user support
- **Email/SMS alerts** for critical violations
- Time-based logging of multiple face incidents
- Integration with exam management systems
- **Web-based dashboard** for monitoring multiple students
- **Cloud-based processing** for distributed monitoring
- **Mobile app** for remote monitoring

## License

This project is open source and available under the MIT License.