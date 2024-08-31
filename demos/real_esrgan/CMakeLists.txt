# -----| cmake version
cmake_minimum_required(VERSION 3.16)

# -----| kuiper_infer project
# project(real_esrgan LANGUAGES C CXX)

# -----| cpp version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(KuiperInfer)

list(APPEND real_esrgan_link ${OpenCV_LIBS})
list(APPEND real_esrgan_include ${OpenCV_INCLUDE_DIRS})

if(KuiperInfer_FOUND)
    message(STATUS "find KuiperInfer package!")
    list(APPEND real_esrgan_link ${KuiperInfer_LIBRARIES})
else()
    message(STATUS "Can't find KuiperInfer package!")
    list(APPEND real_esrgan_link kuiper)
endif()


message(STATUS "KuiperInfer: ${real_esrgan_link}")

add_executable(real_esrgan real_esrgan.cpp ../image_util.hpp ../image_util.cpp)

target_link_libraries(real_esrgan PUBLIC ${real_esrgan_link})
target_include_directories(real_esrgan PUBLIC ${real_esrgan_include})

kuiper_infer_default_properties(real_esrgan)
kuiper_infer_set_runtime_directory(real_esrgan "${PROJECT_BINARY_DIR}/demos")

install(TARGETS real_esrgan DESTINATION ${CMAKE_INSTALL_BINDIR})
