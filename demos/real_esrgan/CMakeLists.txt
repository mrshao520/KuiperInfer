# -----| cmake version
cmake_minimum_required(VERSION 3.16)

# -----| kuiper_infer project
# project(real_esrgan LANGUAGES C CXX)

# -----| cpp version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(KuiperInfer REQUIRED)

message(STATUS "KuiperInfer: ${KuiperInfer_LIBRARIES}")

add_executable(real_esrgan real_esrgan.cpp ../image_util.hpp ../image_util.cpp)

target_link_libraries(real_esrgan PUBLIC ${KuiperInfer_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(real_esrgan PUBLIC ${OpenCV_INCLUDE_DIRS})

kuiper_infer_default_properties(real_esrgan)
kuiper_infer_set_runtime_directory(real_esrgan "${PROJECT_BINARY_DIR}/demos")

install(TARGETS real_esrgan DESTINATION ${CMAKE_INSTALL_BINDIR})
