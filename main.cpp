#include "behaviordetector.h"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

// ============================================================================
// SYSTEM CONSTANTS
// ============================================================================
const string MODEL_PATH       = "/home/admin/train_dms/models/dms_tl.onnx";
const int CAMERA_SOURCE       = 0; 
const int PROCESS_EVERY_N_FRAMES = 3;  // Process 1 frame, skip 2 (Saves 66% CPU)
const int MAX_FRAMES          = 900;   // Auto-terminate after 30 seconds (at 30fps)
const string LOG_FILE         = "train_dms_log.txt";

int main() {
    cout << "TrainDMS - Object Oriented System starting..." << endl;
    cout << "Loading Model: " << MODEL_PATH << endl;

    // Initialize and clear the text log file for a new run
    ofstream logfile(LOG_FILE, ios::trunc);
    logfile << "=== TrainDMS Log started at " << __DATE__ << " " << __TIME__ << " ===\n";
    logfile.close();

    // Instantiate the AI behavior detector object
    BehaviorDetector detector(MODEL_PATH, 640, 0.05f, 0.45f);

    // Open Video4Linux backend (highly optimized for Debian/Ubuntu webcams)
    VideoCapture cap(CAMERA_SOURCE, CAP_V4L2);
    if (!cap.isOpened()) {
        cerr << "CRITICAL ERROR: Could not open camera source." << endl;
        return -1;
    }

    // Retrieve camera resolution and FPS settings
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0.0) fps = 30.0; // Fallback safely if camera fails to report FPS

    // Setup the MP4 Video Writer to save evidence
    string output_video = "dms_output.mp4";
    VideoWriter writer(output_video, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(frame_width, frame_height));

    Mat frame;
    int frame_count = 0;

    // ========================================================================
    // THE INFINITE EVENT LOOP
    // ========================================================================
    while (true) {
        // Read next frame. Break loop if video stream ends or camera disconnects.
        if (!cap.read(frame)) break;
        if (frame.empty()) continue;

        frame_count++;
        
        // Determine if this frame should be sent through the heavy AI network.
        // True on frames 3, 6, 9... False on others.
        bool run_ai = (frame_count % PROCESS_EVERY_N_FRAMES == 0);

        // Pass the frame to the object. It will modify the image in-place (adding boxes/text).
        detector.processFrame(frame, run_ai);

        // Save the annotated frame to disk
        writer.write(frame);
        
        // Print a heartbeat every second to console to prove it hasn't crashed
        if (frame_count % 30 == 0) {
            cout << "Processed " << frame_count << " / " << MAX_FRAMES << " frames..." << endl;
        }

        // Headless graceful exit condition
        if (frame_count >= MAX_FRAMES) break; 
    }

    // Release hardware bindings safely
    cap.release();
    writer.release(); 
    cout << "TrainDMS execution completed safely. Video saved." << endl;
    
    return 0;
}
