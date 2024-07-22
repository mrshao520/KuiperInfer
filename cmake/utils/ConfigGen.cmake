################################################################################################
# Function for generation kuiper_infer build- and install- tree export config files
# 用于生成kuiper_infer构建和安装树导出配置文件的函数
# Usage:
#  kuiper_infer_generate_export_configs()
function(kuiper_infer_generate_export_configs)
    set(install_cmake_suffix "share/KuiperInfer")

    if(NOT HAVE_CUDA)
        set(HAVE_CUDA FALSE)
    endif()

    if(NOT HAVE_CUDNN)
        set(HAVE_CUDNN FALSE)
    endif()

    # -----| Configure build-tree KuiperInferConfig.cmake file
    configure_file("cmake/templates/KuiperInferConfig.cmake.in"
                    "${PROJECT_BINARY_DIR}/KuiperInfer.cmake" @ONLY)         

    # Add targets to the build-tree export set
    # 导出外部项目的目标或包，以便直接从当前项目的构建树中使用，而无需安装
    # export导出构建的目标，可以被其他CMAKE项目通过find_package命令找到
    export(TARGETS kuiper # 指定要导出的目标列表
            # 指定要导出文件的名称
            FILE "${PROJECT_BINARY_DIR}/KuiperInferTargets.cmake")
    # 声明一个包，以便其他项目可以使用find_package()来查找
    export(PACKAGE KuiperInfer)

    # -----| Configure install-tree KuiperInferConfig.cmake file
    configure_file("cmake/templates/KuiperInferConfig.cmake.in"
                    "${PROJECT_BINARY_DIR}/cmake/KuiperInferConfig.cmake" @ONLY)  
    # Install the KuiperInferConfig.cmake and export set to use with install-tree
    install(FILES "${PROJECT_BINARY_DIR}/cmake/KuiperInferConfig.cmake"
            DESTINATION ${install_cmake_suffix})
    install(EXPORT KuiperInferTargets DESTINATION ${install_cmake_suffix})

    # -----| Configure and install version file
    configure_file(cmake/templates/KuiperInferConfigVersion.cmake.in 
                    "${PROJECT_BINARY_DIR}/KuiperInferConfigVersion.cmake" @ONLY)
    install(FILES "${PROJECT_BINARY_DIR}/KuiperInferConfigVersion.cmake" 
            DESTINATION ${install_cmake_suffix})       
endfunction()