# Train Driver Monitoring System (DMS)
Ứng dụng C++ phát hiện hành vi tài xế (buồn ngủ, dùng điện thoại) sử dụng YOLOv8 và OpenCV.
Chạy trên Debian 12 (IPC).

## 🛠 Môi trường phát triển & Huấn luyện (Laptop Win 11)
Để đảm bảo tính tương thích khi chạy trên IPC, các thông số huấn luyện được ghi nhận như sau:
- **Python Version:** 3.11.14 (Khớp hoàn hảo với Debian 12)
- **YOLO Framework:** Ultralytics v8.4.19 (YOLOv10)
- **Conda Environment:** `yolo_v10`
- **Thư viện chính:** `opencv-python`, `torch`, `torchvision`

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Feb__8_05:53:42_Coordinated_Universal_Time_2023
Cuda compilation tools, release 12.1, V12.1.66
Build cuda_12.1.r12.1/compiler.32415258_0

## 🚀 Môi trường triển khai (IPC Debian 12)
- **Ngôn ngữ:** C++ (GCC 12)
- **Framework:** 
- **Thư viện xử lý ảnh:** OpenCV 4.10.0
- **Inference Engine:** OpenCV DNN (sử dụng file .onnx được export từ Laptop)
