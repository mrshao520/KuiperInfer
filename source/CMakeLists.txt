# -----| KuiperInfer library
file(GLOB_RECURSE hdrs ${PROJECT_ROOT}/include/*.h*)
file(GLOB_RECURSE srcs ${PROJECT_ROOT}/source/*.cpp)
file(GLOB_RECURSE cuda ${PROJECT_ROOT}/source/*.cu)

list(APPEND srcs ${hdrs})

add_library(kuiper SHARED ${srcs})
set_target_properties(kuiper PROPERTIES 
    DEBUG_POSTFIX ${KuiperInfer_POSTFIX}
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
    VERSION ${KUIPERINFER_TARGET_VERSION}
    SOVERSION ${KUIPERINFER_TARGET_SOVERSION})

target_link_libraries(kuiper ${KuiperInfer_LINKER_LIBS})
target_include_directories(kuiper ${KuiperInfer_INCLUDE_DIRS}
        PUBLIC
        $<BUILD_INTERFACE:${KuiperInfer_INCLUDE_DIR}> # 构建时头目录
        $<INSTALL_INTERFACE:include>)                 # 安装时头目录
if(KuiperInfer_DEFINITIONS)
    target_compile_definitions(kuiper ${KuiperInfer_DEFINITIONS})
endif()
if(KuiperInfer_COMPILE_OPTIONS)
    target_compile_options(kuiper ${KuiperInfer_COMPILE_OPTIONS})
endif()

# -----| install
install(DIRECTORY ${KuiperInfer_INCLUDE_DIR}/kuiper DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS kuiper EXPORT KuiperInferTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})