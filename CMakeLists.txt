set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_minimum_required(VERSION 2.8)
project( Matching )


find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(/opt/homebrew/Cellar/onnxruntime/1.20.1_2/include/onnxruntime)

link_directories(/opt/homebrew/Cellar/onnxruntime/1.20.1_2/lib)

add_executable( ObjectRecognition src/detect_object.cpp src/helpers.cpp)

target_link_libraries( ObjectRecognition ${OpenCV_LIBS} )


