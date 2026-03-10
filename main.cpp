#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// ============================================================================
// CONFIGURATION PARAMETERS
// ============================================================================
const string MODEL_PATH       = "/home/admin/train_dms/models/dms_tl.onnx";
const int CAMERA_SOURCE       = 0; 
const float  CONF_THRESHOLD   = 0.05f; // Base confidence threshold for initial filtering
const float  NMS_THRESHOLD    = 0.45f; // Non-Maximum Suppression threshold to remove overlapping boxes
const int    INPUT_SIZE       = 640;   // Standard YOLOv8 input resolution (640x640)
const string LOG_FILE         = "train_dms_log.txt";

// --- Optimization and Alarm Logic ---
// Skip frames to dramatically reduce CPU usage (e.g., process 1 out of 3 frames)
const int PROCESS_EVERY_N_FRAMES  = 3; 

// Debounce Logic: Minimum consecutive frames required to trigger an actual alarm.
// This prevents false positives from single-frame glitches (e.g., normal blinking).
const int NO_FACE_ALARM_FRAMES    = 10; // Approx. 3.0 seconds of absence
const int DISTRACTED_ALARM_FRAMES = 7;  // Approx. 2.0 seconds of looking away
const int PHONE_ALARM_FRAMES      = 7;  // Approx. 2.0 seconds of phone usage
const int SLEEPY_ALARM_FRAMES     = 5;  // Approx. 1.5 seconds of closed eyes/yawning
const int SMOKING_ALARM_FRAMES    = 5;  // Approx. 1.5 seconds of smoking

// Standardized list of 7 classes for the Train Driver Monitoring model.
// CRITICAL: The order MUST match the 'names' array in your data.yaml perfectly.
const vector<string> CLASS_NAMES = {
    "eyeclose", "face", "phone", "yawn", "smoking", "distraction", "drowsy"
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Dynamically sets the confidence threshold based on the specific class.
 * Small or ambiguous objects (like closed eyes) need higher sensitivity (lower threshold),
 * while distinct objects (like a face) require higher confidence.
 */
float getClassThreshold(int class_id) {
    switch (class_id) {
        case 0: return 0.20f; // eyeclose: Small target, requires high sensitivity
        case 1: return 0.30f; // face: Distinct feature, demands higher confidence
        case 2: return 0.30f; // phone: Medium target
        case 3: return 0.20f; // yawn: Medium target
        case 4: return 0.20f; // smoking: Small target (cigarette)
        case 5: return 0.30f; // distraction: Behavioral state
        case 6: return 0.25f; // drowsy: Behavioral state
        default: return 0.25f;
    }
}

// Structure to store parsed detection results
struct Detection {
    int class_id;
    float confidence;
    Rect box;
};

/**
 * @brief Logs alarm events to both the console (stdout) and a persistent text file.
 * Includes a timestamp for auditing.
 */
void logAndPrint(const string& behavior, float conf) {
    auto now = chrono::system_clock::now();
    time_t tt = chrono::system_clock::to_time_t(now);
    tm local_tm = *localtime(&tt);
    stringstream ss;
    ss << put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << " | EVENT | " << behavior
       << " | conf: " << fixed << setprecision(3) << conf << endl;
    
    string line = ss.str();
    cout << line; // Print to terminal
    
    // Append to log file
    ofstream logfile(LOG_FILE, ios::app);
    if (logfile.is_open()) {
        logfile << line;
        logfile.close();
    }
}

/**
 * @brief Resizes the input image to a square format required by YOLO without distorting the aspect ratio.
 * It scales the image and adds gray padding (letterboxing) to the remaining areas.
 */
Mat letterbox(const Mat& source, int input_size) {
    float scale = min((float)input_size / source.cols, (float)input_size / source.rows);
    int new_w = source.cols * scale;
    int new_h = source.rows * scale;

    Mat resized;
    resize(source, resized, Size(new_w, new_h));

    // Calculate padding for centering
    int pad_w = input_size - new_w;
    int pad_h = input_size - new_h;

    int top = pad_h / 2;
    int bottom = pad_h - top;
    int left = pad_w / 2;
    int right = pad_w - left;

    Mat result;
    // Add gray borders (114, 114, 114 is standard for YOLO)
    copyMakeBorder(resized, result, top, bottom, left, right, BORDER_CONSTANT, Scalar(114, 114, 114));
    return result;
}

/**
 * @brief Parses the raw output tensor from the YOLOv8 ONNX model and converts it into physical bounding boxes.
 */
vector<Detection> postprocess(const vector<Mat>& outputs, const Size& frame_size,
                              float conf_thres, float nms_thres) {
    vector<Detection> detections;
    if (outputs.empty()) return detections;

    Mat output = outputs[0];
    
    // YOLOv8 outputs a 3D tensor [batch_size, channels, features]. 
    // We flatten and transpose it to a 2D matrix [features, channels] for easier parsing.
    if (output.dims == 3) {
        Mat output2d(output.size[1], output.size[2], CV_32F, output.ptr<float>());
        Mat transposed; 
        cv::transpose(output2d, transposed); 
        output = transposed; 
    }

    // Number of classes = total columns - 4 coordinate values (cx, cy, w, h)
    int num_classes = output.cols - 4;
    vector<Rect> boxes;
    vector<float> confidences;
    vector<int> class_ids;

    // Recalculate scaling factors to map coordinates back from 640x640 to original frame size
    float scale = min((float)INPUT_SIZE / frame_size.width, (float)INPUT_SIZE / frame_size.height);
    int pad_w = (INPUT_SIZE - frame_size.width * scale) / 2;
    int pad_h = (INPUT_SIZE - frame_size.height * scale) / 2;

    // Iterate through all predictions
    for (int i = 0; i < output.rows; ++i) {
        Mat row = output.row(i);
        Mat scores = row.colRange(4, 4 + num_classes);

        Point class_id_pt;
        double max_class_score;
        // Find the class with the highest probability
        minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id_pt);

        float obj_conf = static_cast<float>(max_class_score);
        
        // Discard low-confidence predictions using class-specific thresholds
        if (obj_conf > getClassThreshold(class_id_pt.x)) {
            // YOLO outputs Center X, Center Y, Width, Height
            float cx = row.at<float>(0);
            float cy = row.at<float>(1);
            float w  = row.at<float>(2);
            float h  = row.at<float>(3);

            // Convert center coordinates to top-left corner and remove padding mapping
            int left   = static_cast<int>((cx - w / 2.0f - pad_w) / scale);
            int top    = static_cast<int>((cy - h / 2.0f - pad_h) / scale);
            int width  = static_cast<int>(w / scale);
            int height = static_cast<int>(h / scale);

            boxes.emplace_back(left, top, width, height);
            confidences.push_back(obj_conf);
            class_ids.push_back(class_id_pt.x);
        }
    }

    // Apply Non-Maximum Suppression (NMS) to eliminate redundant, overlapping boxes for the same object
    vector<Rect> nms_boxes;
    int max_wh = 4096; // Offset strategy for multi-class NMS
    for (size_t i = 0; i < boxes.size(); i++) {
        int offset = class_ids[i] * max_wh;
        nms_boxes.push_back(Rect(boxes[i].x + offset, boxes[i].y + offset, boxes[i].width, boxes[i].height));
    }

    vector<int> indices;
    dnn::NMSBoxes(nms_boxes, confidences, conf_thres, nms_thres, indices);

    // Save final valid detections
    for (int idx : indices) {
        if (class_ids[idx] >= 0 && class_ids[idx] < static_cast<int>(CLASS_NAMES.size())) {
            detections.push_back({class_ids[idx], confidences[idx], boxes[idx]});
        }
    }
    return detections;
}

// ============================================================================
// MAIN EXECUTION LOOP
// ============================================================================
int main() {
    cout << "TrainDMS - Driver Monitoring System starting in Headless Mode..." << endl;
    cout << "Loading Model: " << MODEL_PATH << endl;

    // Initialize/clear the log file
    ofstream logfile(LOG_FILE, ios::trunc);
    logfile << "=== TrainDMS Log started at " << __DATE__ << " " << __TIME__ << " ===\n";
    logfile.close();

    // Load the ONNX model using OpenCV DNN
    Net net = readNet(MODEL_PATH);

    // Initialize Video Capture (V4L2 backend is optimized for Linux/Debian)
    VideoCapture cap(CAMERA_SOURCE, CAP_V4L2);
    if (!cap.isOpened()) {
        cerr << "CRITICAL ERROR: Could not open camera source." << endl;
        return -1;
    }

    // Retrieve camera properties
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0.0) fps = 30.0; // Fallback FPS if camera fails to report it

    // Setup VideoWriter to save the output locally for reviewing
    string output_video = "dms_output.mp4";
    VideoWriter writer(output_video, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(frame_width, frame_height));

    Mat frame;
    int frame_count = 0;
    
    // Limit execution length. Program will safely terminate and save video after this limit.
    // 600 frames at 30fps = 20 seconds of recording.
    const int MAX_FRAMES = 600; 

    // State machine variables for Debounce Logic
    int consecutive_no_face = 0;    bool is_no_face = false;
    int consecutive_distracted = 0; bool is_distracted = false;
    int consecutive_phone = 0;      bool is_using_phone = false;
    int consecutive_sleepy = 0;     bool is_sleepy = false;
    int consecutive_smoking = 0;    bool is_smoking = false;

    vector<Detection> last_detections; 

    // Main frame processing loop
    while (true) {
        if (!cap.read(frame)) break;
        if (frame.empty()) continue;

        frame_count++;

        // --- STEP 1: AI INFERENCE ---
        // Only run the heavy AI inference every N frames to save CPU resources
        if (frame_count % PROCESS_EVERY_N_FRAMES == 0) {
            
            // Preprocess frame: Letterbox -> Create Blob -> Normalize (1/255.0)
            Mat letterboxed_frame = letterbox(frame, INPUT_SIZE);
            Mat blob = blobFromImage(letterboxed_frame, 1.0 / 255.0, Size(INPUT_SIZE, INPUT_SIZE),
                                     Scalar(0,0,0), true, false);

            net.setInput(blob);
            vector<Mat> outputs;
            // Forward pass (Execute the Neural Network)
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            // Extract bounding boxes
            last_detections = postprocess(outputs, frame.size(), CONF_THRESHOLD, NMS_THRESHOLD);

            // Reset current frame flags
            bool found_driver = false; 
            bool found_distracted = false;
            bool found_phone = false;
            bool found_sleepy = false;
            bool found_smoking = false;

            for (const auto& det : last_detections) {
                const string& behavior = CLASS_NAMES[det.class_id];
                
                // If ANY valid class is detected, a driver is present
                found_driver = true;

                // Categorize behaviors
                if (behavior == "distraction") found_distracted = true;
                if (behavior == "phone") found_phone = true;
                if (behavior == "smoking") found_smoking = true;
                // Group micro-expressions into a general 'Sleepy' state
                if (behavior == "eyeclose" || behavior == "yawn" || behavior == "drowsy") {
                    found_sleepy = true;
                }
            }

            // --- STEP 2: TRAIN DRIVER ALARM LOGIC (DEBOUNCE/FILTERING) ---
            
            // 2.1 Missing Driver Alarm
            if (!found_driver) {
                consecutive_no_face++;
                if (consecutive_no_face >= NO_FACE_ALARM_FRAMES && !is_no_face) {
                    logAndPrint("ALARM TRIGGERED: NO DRIVER DETECTED", 1.0);
                    is_no_face = true;
                }
            } else {
                consecutive_no_face = 0;
                is_no_face = false;
            }

            // 2.2 Distraction Alarm
            if (found_distracted) {
                consecutive_distracted++;
                if (consecutive_distracted >= DISTRACTED_ALARM_FRAMES && !is_distracted) {
                    logAndPrint("ALARM TRIGGERED: DRIVER DISTRACTED", 1.0);
                    is_distracted = true;
                }
            } else {
                consecutive_distracted = 0;
                is_distracted = false;
            }

            // 2.3 Phone Usage Alarm
            if (found_phone) {
                consecutive_phone++;
                if (consecutive_phone >= PHONE_ALARM_FRAMES && !is_using_phone) {
                    logAndPrint("ALARM TRIGGERED: ILLEGAL PHONE USAGE", 1.0);
                    is_using_phone = true;
                }
            } else {
                consecutive_phone = 0;
                is_using_phone = false;
            }

            // 2.4 Drowsiness/Fatigue Alarm
            if (found_sleepy) {
                consecutive_sleepy++;
                if (consecutive_sleepy >= SLEEPY_ALARM_FRAMES && !is_sleepy) {
                    logAndPrint("ALARM TRIGGERED: CRITICAL - DROWSY/SLEEPING", 1.0);
                    is_sleepy = true;
                }
            } else {
                consecutive_sleepy = 0;
                is_sleepy = false;
            }

            // 2.5 Smoking Alarm
            if (found_smoking) {
                consecutive_smoking++;
                if (consecutive_smoking >= SMOKING_ALARM_FRAMES && !is_smoking) {
                    logAndPrint("ALARM TRIGGERED: SMOKING IN CABIN", 1.0);
                    is_smoking = true;
                }
            } else {
                consecutive_smoking = 0;
                is_smoking = false;
            }
        }

        // --- STEP 3: RENDER VISUALS (Bounding Boxes & Warning Texts) ---
        // Render bounding boxes from the last inference pass
        for (const auto& det : last_detections) {
            const string& behavior = CLASS_NAMES[det.class_id];
            rectangle(frame, det.box, Scalar(0, 255, 0), 2);
            string label = behavior + " " + to_string(det.confidence).substr(0, 4);
            putText(frame, label, Point(det.box.x, det.box.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }

        // Render Warning Texts dynamically based on active alarms
        int y_pos = 50; // Starting Y coordinate for text
        if (is_no_face) {
            putText(frame, "WARNING: NO DRIVER!!!", Point(30, y_pos), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 255), 3);
            y_pos += 40; // Shift down for next message
        }
        if (is_distracted) {
            putText(frame, "WARNING: DISTRACTED!!!", Point(30, y_pos), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 165, 255), 3);
            y_pos += 40;
        }
        if (is_using_phone) {
            putText(frame, "WARNING: PHONE USAGE!!!", Point(30, y_pos), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 0, 0), 3);
            y_pos += 40;
        }
        if (is_sleepy) {
            putText(frame, "WARNING: DROWSY/SLEEPING!!!", Point(30, y_pos), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 255), 3);
            y_pos += 40;
        }
        if (is_smoking) {
            putText(frame, "WARNING: SMOKING!!!", Point(30, y_pos), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 255, 255), 3);
        }

        // Save the annotated frame to the output video file
        writer.write(frame);
        
        // --- STEP 4: SYSTEM MONITORING ---
        // Print a heartbeat message every 30 frames (~1 sec) to verify the process hasn't frozen
        if (frame_count % 30 == 0) {
            cout << "Processed " << frame_count << " / " << MAX_FRAMES << " frames..." << endl;
        }

        // Auto-terminate condition for Headless Mode
        if (frame_count >= MAX_FRAMES) {
            break; 
        }
    }

    // --- STEP 5: CLEANUP AND EXPORT ---
    // Safely release hardware resources to prevent memory leaks or corrupted video files
    cap.release();
    writer.release(); 
    cout << "TrainDMS execution completed safely. Output video saved as " << output_video << endl;
    
    return 0;
}