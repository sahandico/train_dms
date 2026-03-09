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
// CONFIGURATION
// ============================================================================
const string MODEL_PATH       = "/home/admin/train_dms/models/dms_tauhoa_best.onnx"; 
const int CAMERA_SOURCE       = 0; 
const float  CONF_THRESHOLD   = 0.05f; // Base confidence threshold
const float  NMS_THRESHOLD    = 0.45f; // Non-Maximum Suppression threshold to remove overlapping boxes
const int    INPUT_SIZE       = 640;   // YOLOv8 input size (must match your training imgsz)
const string LOG_FILE         = "train_dms_log.txt";

// --- Optimization & Event Logic Thresholds ---
// PROCESS_EVERY_N_FRAMES: Skips frames to reduce CPU/IPC load. 
// e.g., Set to 3 means AI runs 10 times per second on a 30fps camera.
const int PROCESS_EVERY_N_FRAMES  = 3; 

// Alarm frame thresholds (debounce logic). 
// An event must be detected consecutively for N frames to trigger an alarm.
const int NO_FACE_ALARM_FRAMES    = 10; // ~3.0 seconds (if processing 10fps)
const int DISTRACTED_ALARM_FRAMES = 7;  // ~2.0 seconds
const int PHONE_ALARM_FRAMES      = 7;  // ~2.0 seconds
const int SLEEPY_ALARM_FRAMES     = 5;  // ~1.5 seconds
const int SMOKING_ALARM_FRAMES    = 5;  // ~1.5 seconds

// The 7 classes defined in your data.yaml. 
// THE ORDER MUST MATCH EXACTLY THE IDS FROM 0 TO 6.
const vector<string> CLASS_NAMES = {
    "eyeclose", "face", "phone", "yawn", "smoking", "distraction", "drowsy"
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Dynamically adjust confidence thresholds based on the specific class.
// This helps to filter out false positives for difficult classes.
float getClassThreshold(int class_id) {
    switch (class_id) {
        case 0: return 0.05f; // eyeclose: Tiny object, lower threshold to avoid missing it
        case 1: return 0.35f; // face: Large and distinct, require higher confidence
        case 2: return 0.30f; // phone: Medium confidence
        case 3: return 0.20f; // yawn: Medium/Low to capture mouth states
        case 4: return 0.25f; // smoking: Small object (cigarette)
        case 5: return 0.30f; // distraction: Looking away/doing other things
        case 6: return 0.25f; // drowsy: Nodding off
        default: return 0.25f;
    }
}

// Struct to store bounding box and classification info
struct Detection {
    int class_id;
    float confidence;
    Rect box;
};

// Helper function to log events to console and a text file
void logAndPrint(const string& behavior, float conf) {
    auto now = chrono::system_clock::now();
    time_t tt = chrono::system_clock::to_time_t(now);
    tm local_tm = *localtime(&tt);
    stringstream ss;
    ss << put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << " | EVENT | " << behavior
       << " | conf: " << fixed << setprecision(3) << conf << endl;
    string line = ss.str();
    cout << line;
    ofstream logfile(LOG_FILE, ios::app);
    if (logfile.is_open()) {
        logfile << line;
        logfile.close();
    }
}

// Letterbox resize algorithm required by YOLOv8.
// It resizes the image while keeping the aspect ratio by adding gray padding.
Mat letterbox(const Mat& source, int input_size) {
    float scale = min((float)input_size / source.cols, (float)input_size / source.rows);
    int new_w = source.cols * scale;
    int new_h = source.rows * scale;

    Mat resized;
    resize(source, resized, Size(new_w, new_h));

    int pad_w = input_size - new_w;
    int pad_h = input_size - new_h;

    int top = pad_h / 2;
    int bottom = pad_h - top;
    int left = pad_w / 2;
    int right = pad_w - left;

    Mat result;
    // Fill padding with neutral gray (114, 114, 114)
    copyMakeBorder(resized, result, top, bottom, left, right, BORDER_CONSTANT, Scalar(114, 114, 114));
    return result;
}

// Parses the raw tensor output from YOLOv8 into usable bounding boxes
vector<Detection> postprocess(const vector<Mat>& outputs, const Size& frame_size,
                              float conf_thres, float nms_thres) {
    vector<Detection> detections;
    if (outputs.empty()) return detections;

    Mat output = outputs[0];
    // YOLOv8 output tensor is typically [1, 4 + num_classes, 8400]
    // We transpose it to [8400, 4 + num_classes] for easier row-by-row reading
    if (output.dims == 3) {
        Mat output2d(output.size[1], output.size[2], CV_32F, output.ptr<float>());
        Mat transposed; 
        cv::transpose(output2d, transposed); 
        output = transposed; 
    }

    int num_classes = output.cols - 4;
    vector<Rect> boxes;
    vector<float> confidences;
    vector<int> class_ids;

    // Calculate scale and padding used during letterbox to map coordinates back to the original frame
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
        // Compare against the class-specific threshold
        if (obj_conf > getClassThreshold(class_id_pt.x)) {
            // Extract bounding box coordinates (Center X, Center Y, Width, Height)
            float cx = row.at<float>(0);
            float cy = row.at<float>(1);
            float w  = row.at<float>(2);
            float h  = row.at<float>(3);

            // Convert to Top-Left X, Top-Left Y and map back to original image scale
            int left   = static_cast<int>((cx - w / 2.0f - pad_w) / scale);
            int top    = static_cast<int>((cy - h / 2.0f - pad_h) / scale);
            int width  = static_cast<int>(w / scale);
            int height = static_cast<int>(h / scale);

            boxes.emplace_back(left, top, width, height);
            confidences.push_back(obj_conf);
            class_ids.push_back(class_id_pt.x);
        }
    }

    // Apply Non-Maximum Suppression (NMS) to remove overlapping boxes for the same object
    vector<Rect> nms_boxes;
    int max_wh = 4096; // Offset trick to perform independent NMS per class
    for (size_t i = 0; i < boxes.size(); i++) {
        int offset = class_ids[i] * max_wh;
        nms_boxes.push_back(Rect(boxes[i].x + offset, boxes[i].y + offset, boxes[i].width, boxes[i].height));
    }

    vector<int> indices;
    dnn::NMSBoxes(nms_boxes, confidences, conf_thres, nms_thres, indices);

    // Save the final filtered detections
    for (int idx : indices) {
        if (class_ids[idx] >= 0 && class_ids[idx] < static_cast<int>(CLASS_NAMES.size())) {
            detections.push_back({class_ids[idx], confidences[idx], boxes[idx]});
        }
    }
    return detections;
}

// ============================================================================
// MAIN LOOP
// ============================================================================
int main() {
    cout << "TrainDMS - Driver Monitoring System starting..." << endl;
    cout << "Model: " << MODEL_PATH << endl;

    // Initialize Log File
    ofstream logfile(LOG_FILE, ios::trunc);
    logfile << "=== TrainDMS Log started at " << __DATE__ << " " << __TIME__ << " ===\n";
    logfile.close();

    // Load YOLOv8 ONNX model via OpenCV DNN
    Net net = readNet(MODEL_PATH);

    // Initialize Camera (V4L2 for Linux, remove CAP_V4L2 if on Windows)
    VideoCapture cap(CAMERA_SOURCE, CAP_V4L2);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera source." << endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0.0) fps = 30.0; 

    // Setup Video Writer to save the output for review
    string output_video = "dms_output.mp4";
    VideoWriter writer(output_video, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(frame_width, frame_height));

    Mat frame;
    int frame_count = 0;

    // --- State Variables for Event Logic (Counters and Flags) ---
    int consecutive_no_face = 0;    bool is_no_face = false;
    int consecutive_distracted = 0; bool is_distracted = false;
    int consecutive_phone = 0;      bool is_using_phone = false;
    int consecutive_sleepy = 0;     bool is_sleepy = false;
    int consecutive_smoking = 0;    bool is_smoking = false;

    vector<Detection> last_detections; 

    // Main processing loop
    while (true) {
        if (!cap.read(frame)) break;
        if (frame.empty()) continue;

        frame_count++;

        // Step 1: Frame Skipping - Only process 1 out of every N frames
        if (frame_count % PROCESS_EVERY_N_FRAMES == 0) {
            
            // Step 2: Preprocessing and DNN Forward Pass
            Mat letterboxed_frame = letterbox(frame, INPUT_SIZE);
            Mat blob = blobFromImage(letterboxed_frame, 1.0 / 255.0, Size(INPUT_SIZE, INPUT_SIZE),
                                     Scalar(0,0,0), true, false);

            net.setInput(blob);
            vector<Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            // Step 3: Postprocessing (Decode boxes, apply NMS)
            last_detections = postprocess(outputs, frame.size(), CONF_THRESHOLD, NMS_THRESHOLD);

            // Step 4: Analyze Current Frame States
            bool found_driver = false; 
            bool found_distracted = false;
            bool found_phone = false;
            bool found_sleepy = false;
            bool found_smoking = false;

            for (const auto& det : last_detections) {
                const string& behavior = CLASS_NAMES[det.class_id];
                
                // If the model detects ANY of the 7 classes, it implies someone is sitting there
                found_driver = true;

                // Map raw YOLO classes to Railway DMS Events
                if (behavior == "distraction") found_distracted = true;
                if (behavior == "phone") found_phone = true;
                if (behavior == "smoking") found_smoking = true;
                
                // Combine 'eyeclose', 'yawn', and 'drowsy' into a single Sleepy event
                if (behavior == "eyeclose" || behavior == "yawn" || behavior == "drowsy") {
                    found_sleepy = true;
                }
            }

            // Step 5: Execute Debounce Logic (Count consecutive frames)
            
            // --- EVENT 1: NO DRIVER ---
            if (!found_driver) {
                consecutive_no_face++;
                if (consecutive_no_face >= NO_FACE_ALARM_FRAMES && !is_no_face) {
                    logAndPrint("ALARM TRIGGERED: NO DRIVER", 1.0);
                    is_no_face = true;
                }
            } else {
                consecutive_no_face = 0; // Reset counter immediately if driver appears
                is_no_face = false;
            }

            // --- EVENT 2: DISTRACTED ---
            if (found_distracted) {
                consecutive_distracted++;
                if (consecutive_distracted >= DISTRACTED_ALARM_FRAMES && !is_distracted) {
                    logAndPrint("ALARM TRIGGERED: DISTRACTED", 1.0);
                    is_distracted = true;
                }
            } else {
                consecutive_distracted = 0;
                is_distracted = false;
            }

            // --- EVENT 3: PHONE USAGE ---
            if (found_phone) {
                consecutive_phone++;
                if (consecutive_phone >= PHONE_ALARM_FRAMES && !is_using_phone) {
                    logAndPrint("ALARM TRIGGERED: PHONE USAGE", 1.0);
                    is_using_phone = true;
                }
            } else {
                consecutive_phone = 0;
                is_using_phone = false;
            }

            // --- EVENT 4: DROWSY / SLEEPY ---
            if (found_sleepy) {
                consecutive_sleepy++;
                if (consecutive_sleepy >= SLEEPY_ALARM_FRAMES && !is_sleepy) {
                    logAndPrint("ALARM TRIGGERED: DROWSY/SLEEPING", 1.0);
                    is_sleepy = true;
                }
            } else {
                consecutive_sleepy = 0;
                is_sleepy = false;
            }

            // --- EVENT 5: SMOKING ---
            if (found_smoking) {
                consecutive_smoking++;
                if (consecutive_smoking >= SMOKING_ALARM_FRAMES && !is_smoking) {
                    logAndPrint("ALARM TRIGGERED: SMOKING", 1.0);
                    is_smoking = true;
                }
            } else {
                consecutive_smoking = 0;
                is_smoking = false;
            }
        }

        // Step 6: Visualization
        // Draw bounding boxes for all detected objects (persists across skipped frames)
        for (const auto& det : last_detections) {
            const string& behavior = CLASS_NAMES[det.class_id];
            rectangle(frame, det.box, Scalar(0, 255, 0), 2);
            string label = behavior + " " + to_string(det.confidence).substr(0, 4);
            putText(frame, label, Point(det.box.x, det.box.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }

        // Draw active Alarms on screen (Stack them vertically to prevent text overlap)
        int y_pos = 50;
        if (is_no_face) {
            putText(frame, "WARNING: NO DRIVER!!!", Point(30, y_pos), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 255), 3);
            y_pos += 40;
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

        // Save and Display
        writer.write(frame);
        imshow("DMS Railway Cabin", frame);
        
        // Press ESC to exit gracefully
        if (waitKey(1) == 27) break; 
    }

    // Cleanup
    cap.release();
    writer.release(); 
    destroyAllWindows();
    cout << "TrainDMS stopped gracefully. Video saved." << endl;
    return 0;
}