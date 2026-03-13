cmake_minimum_required(VERSION 3.10)
project(TrainDMS CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Nếu bạn cài OpenCV 4.10 từ source (thường nằm trong /usr/local), hãy set cứng thư mục:
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4") # Thay đổi đường dẫn này theo thực tế máy bạn

# Bắt buộc phải tìm thấy bản 4.10 trở lên
find_package(OpenCV 4.10 REQUIRED) 
find_package(Threads REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}") # In ra để kiểm tra lúc chạy cmake

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(TrainDMS main.cpp behaviordetector.cpp)

target_link_libraries(TrainDMS ${OpenCV_LIBS} Threads::Threads)