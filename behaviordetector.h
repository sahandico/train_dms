#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

// ============================================================================
// DATA STRUCTURES
// ============================================================================
/**
 * @brief Structure to store parsed detection results from the YOLO model.
 * Holds the predicted class, confidence score, and physical bounding box.
 */
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

// ============================================================================
// CLASS DEFINITION: BehaviorDetector
// ============================================================================
/**
 * @brief Encapsulates the AI Inference and Alarm Logic for Train Driver Monitoring.
 * Handles image preprocessing, ONNX model execution, postprocessing (NMS),
 * and state-machine-based alarm triggering.
 */
class BehaviorDetector {
private:
    // Core AI Components
    cv::dnn::Net net;                        // OpenCV DNN Network object
    int input_size;                          // Standard YOLO input dimension (e.g., 640)
    float conf_threshold;                    // Base confidence to filter weak predictions
    float nms_threshold;                     // Threshold for Non-Maximum Suppression
    std::vector<std::string> class_names;    // Alphabetically ordered class labels
    std::string log_file;                    // Path to save event logs

    // --- State Machine Variables for Debounce Logic ---
    // Tracks consecutive frames of a specific behavior to prevent false alarms
    int consecutive_no_face = 0;    bool is_no_face = false;
    int consecutive_distracted = 0; bool is_distracted = false;
    int consecutive_phone = 0;      bool is_using_phone = false;
    int consecutive_sleepy = 0;     bool is_sleepy = false;
    int consecutive_smoking = 0;    bool is_smoking = false;

    // Stores the last valid detections to draw on skipped frames (to save CPU)
    std::vector<Detection> last_detections;

    // --- Debounce Thresholds (Frames) ---
    // Minimum continuous frames required to transition to an ALARM state.
    const int NO_FACE_ALARM_FRAMES    = 10; // ~3.0s (assuming processing 1/3 frames at 10fps)
    const int DISTRACTED_ALARM_FRAMES = 7;  // ~2.0s
    const int PHONE_ALARM_FRAMES      = 7;  // ~2.0s
    const int SLEEPY_ALARM_FRAMES     = 5;  // ~1.5s
    const int SMOKING_ALARM_FRAMES    = 5;  // ~1.5s

    // --- Internal Helper Methods ---
    float getClassThreshold(int class_id);
    void logAndPrint(const std::string& behavior, float conf);
    cv::Mat letterbox(const cv::Mat& source);
    std::vector<Detection> postprocess(const std::vector<cv::Mat>& outputs, const cv::Size& frame_size);
    void updateAlarmLogic(const std::vector<Detection>& detections);
    void renderVisuals(cv::Mat& frame);

public:
    /**
     * @brief Initializes the detector with the specified ONNX model.
     * @param model_path Absolute path to the .onnx weights file.
     * @param size The input resolution expected by the model (default 640).
     */
    BehaviorDetector(const std::string& model_path, int size = 640, float conf_th = 0.05f, float nms_th = 0.45f);
    
    /**
     * @brief The main pipeline. Processes a single video frame.
     * @param frame The raw BGR image matrix from the camera.
     * @param run_inference If true, runs the heavy AI model. If false, skips AI and only draws old boxes.
     */
    void processFrame(cv::Mat& frame, bool run_inference);
};
