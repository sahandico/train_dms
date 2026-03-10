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
const string MODEL_PATH       = "/home/admin/train_dms/models/dms_tl.onnx";
const int CAMERA_SOURCE       = 0; 
const float  CONF_THRESHOLD   = 0.05f; // Ngưỡng confidence cơ bản
const float  NMS_THRESHOLD    = 0.45f; // Lọc các khung hình bị trùng
const int    INPUT_SIZE       = 640;   // Kích thước đầu vào chuẩn của YOLOv8 model này
const string LOG_FILE         = "train_dms_log.txt";

// --- Tối ưu và Logic Cảnh báo ---
const int PROCESS_EVERY_N_FRAMES  = 3; // Chỉ chạy AI 1 lần mỗi 3 frame (Giảm tải CPU)

// Số frame liên tiếp để kích hoạt cảnh báo (Debounce logic)
const int NO_FACE_ALARM_FRAMES    = 10; // ~3.0 giây thực tế
const int DISTRACTED_ALARM_FRAMES = 7;  // ~2.0 giây thực tế
const int PHONE_ALARM_FRAMES      = 7;  // ~2.0 giây thực tế
const int SLEEPY_ALARM_FRAMES     = 5;  // ~1.5 giây thực tế
const int SMOKING_ALARM_FRAMES    = 5;  // ~1.5 giây thực tế

// Danh sách 7 class chuẩn của model Lái Tàu Hỏa (Thứ tự phải khớp 100% file data.yaml)
const vector<string> CLASS_NAMES = {
    "eyeclose", "face", "phone", "yawn", "smoking", "distraction", "drowsy"
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Thiết lập ngưỡng Confidence riêng cho từng class để chống báo động giả
float getClassThreshold(int class_id) {
    switch (class_id) {
        case 0: return 0.20f; // eyeclose: Nhỏ, cần điểm nhạy
        case 1: return 0.30f; // face: Dễ nhận, yêu cầu tự tin cao
        case 2: return 0.30f; // phone: Vừa phải
        case 3: return 0.20f; // yawn: Vừa phải
        case 4: return 0.20f; // smoking: Điếu thuốc nhỏ
        case 5: return 0.30f; // distraction: Hành vi không tập trung
        case 6: return 0.25f; // drowsy: Ngủ gật
        default: return 0.25f;
    }
}

struct Detection {
    int class_id;
    float confidence;
    Rect box;
};

// Hàm ghi log ra màn hình và file txt
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

// Thuật toán Letterbox để resize ảnh vuông cho YOLO mà không làm méo tỷ lệ
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
    copyMakeBorder(resized, result, top, bottom, left, right, BORDER_CONSTANT, Scalar(114, 114, 114));
    return result;
}

// Bóc tách kết quả Tensor của YOLOv8 ra thành tọa độ thực tế
vector<Detection> postprocess(const vector<Mat>& outputs, const Size& frame_size,
                              float conf_thres, float nms_thres) {
    vector<Detection> detections;
    if (outputs.empty()) return detections;

    Mat output = outputs[0];
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

    float scale = min((float)INPUT_SIZE / frame_size.width, (float)INPUT_SIZE / frame_size.height);
    int pad_w = (INPUT_SIZE - frame_size.width * scale) / 2;
    int pad_h = (INPUT_SIZE - frame_size.height * scale) / 2;

    for (int i = 0; i < output.rows; ++i) {
        Mat row = output.row(i);
        Mat scores = row.colRange(4, 4 + num_classes);

        Point class_id_pt;
        double max_class_score;
        minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id_pt);

        float obj_conf = static_cast<float>(max_class_score);
        if (obj_conf > getClassThreshold(class_id_pt.x)) {
            float cx = row.at<float>(0);
            float cy = row.at<float>(1);
            float w  = row.at<float>(2);
            float h  = row.at<float>(3);

            int left   = static_cast<int>((cx - w / 2.0f - pad_w) / scale);
            int top    = static_cast<int>((cy - h / 2.0f - pad_h) / scale);
            int width  = static_cast<int>(w / scale);
            int height = static_cast<int>(h / scale);

            boxes.emplace_back(left, top, width, height);
            confidences.push_back(obj_conf);
            class_ids.push_back(class_id_pt.x);
        }
    }

    vector<Rect> nms_boxes;
    int max_wh = 4096; 
    for (size_t i = 0; i < boxes.size(); i++) {
        int offset = class_ids[i] * max_wh;
        nms_boxes.push_back(Rect(boxes[i].x + offset, boxes[i].y + offset, boxes[i].width, boxes[i].height));
    }

    vector<int> indices;
    dnn::NMSBoxes(nms_boxes, confidences, conf_thres, nms_thres, indices);

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
    cout << "TrainDMS - Driver Monitoring System starting in Headless Mode..." << endl;
    cout << "Model: " << MODEL_PATH << endl;

    ofstream logfile(LOG_FILE, ios::trunc);
    logfile << "=== TrainDMS Log started at " << __DATE__ << " " << __TIME__ << " ===\n";
    logfile.close();

    Net net = readNet(MODEL_PATH);

    VideoCapture cap(CAMERA_SOURCE, CAP_V4L2);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera source." << endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0.0) fps = 30.0; 

    string output_video = "dms_output.mp4";
    VideoWriter writer(output_video, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(frame_width, frame_height));

    Mat frame;
    int frame_count = 0;
    
    // ĐẶT GIỚI HẠN SỐ FRAME ĐỂ CHƯƠNG TRÌNH TỰ ĐỘNG DỪNG VÀ LƯU VIDEO THÀNH CÔNG
    // 300 frames ~ 10 giây. Bác có thể tăng lên 900 (30 giây) hoặc tùy ý.
    const int MAX_FRAMES = 600; 

    // Các biến trạng thái đếm khung hình
    int consecutive_no_face = 0;    bool is_no_face = false;
    int consecutive_distracted = 0; bool is_distracted = false;
    int consecutive_phone = 0;      bool is_using_phone = false;
    int consecutive_sleepy = 0;     bool is_sleepy = false;
    int consecutive_smoking = 0;    bool is_smoking = false;

    vector<Detection> last_detections; 

    while (true) {
        if (!cap.read(frame)) break;
        if (frame.empty()) continue;

        frame_count++;

        // --- BƯỚC 1: XỬ LÝ AI ---
        if (frame_count % PROCESS_EVERY_N_FRAMES == 0) {
            
            Mat letterboxed_frame = letterbox(frame, INPUT_SIZE);
            Mat blob = blobFromImage(letterboxed_frame, 1.0 / 255.0, Size(INPUT_SIZE, INPUT_SIZE),
                                     Scalar(0,0,0), true, false);

            net.setInput(blob);
            vector<Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            last_detections = postprocess(outputs, frame.size(), CONF_THRESHOLD, NMS_THRESHOLD);

            bool found_driver = false; 
            bool found_distracted = false;
            bool found_phone = false;
            bool found_sleepy = false;
            bool found_smoking = false;

            for (const auto& det : last_detections) {
                const string& behavior = CLASS_NAMES[det.class_id];
                
                // Thấy bất kỳ nhãn nào -> Khẳng định có người
                found_driver = true;

                // Phân loại sự kiện
                if (behavior == "distraction") found_distracted = true;
                if (behavior == "phone") found_phone = true;
                if (behavior == "smoking") found_smoking = true;
                if (behavior == "eyeclose" || behavior == "yawn" || behavior == "drowsy") {
                    found_sleepy = true;
                }
            }

            // --- BƯỚC 2: LOGIC CẢNH BÁO TÀU HỎA ---
            
            // 1. Không có người lái
            if (!found_driver) {
                consecutive_no_face++;
                if (consecutive_no_face >= NO_FACE_ALARM_FRAMES && !is_no_face) {
                    logAndPrint("ALARM TRIGGERED: NO DRIVER", 1.0);
                    is_no_face = true;
                }
            } else {
                consecutive_no_face = 0;
                is_no_face = false;
            }

            // 2. Không tập trung
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

            // 3. Nghe điện thoại
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

            // 4. Buồn ngủ / Ngáp
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

            // 5. Hút thuốc
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

        // --- BƯỚC 3: VẼ KHUNG LÊN VIDEO ĐỂ LƯU LẠI ---
        for (const auto& det : last_detections) {
            const string& behavior = CLASS_NAMES[det.class_id];
            rectangle(frame, det.box, Scalar(0, 255, 0), 2);
            string label = behavior + " " + to_string(det.confidence).substr(0, 4);
            putText(frame, label, Point(det.box.x, det.box.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }

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

        writer.write(frame);
        
        // --- BƯỚC 4: THEO DÕI VÀ DỪNG AN TOÀN ---
        // In log ra Terminal mỗi 30 frame (~ 1 giây) để biết máy đang không bị treo
        if (frame_count % 30 == 0) {
            cout << "Processed " << frame_count << " / " << MAX_FRAMES << " frames..." << endl;
        }

        // Tự động ngắt vòng lặp khi đạt đủ số lượng MAX_FRAMES
        if (frame_count >= MAX_FRAMES) {
            break; 
        }
    }

    // --- BƯỚC 5: CLEANUP VÀ XUẤT FILE ---
    cap.release();
    writer.release(); 
    cout << "TrainDMS stopped gracefully. Video saved successfully." << endl;
    return 0;
}