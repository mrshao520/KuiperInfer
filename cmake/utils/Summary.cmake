################################################################################################
# KuiperInfer status report function.
# Automatically align right column and selects text based on condition.
# Usage:
#   kuiper_infer_status(<text>)
#   kuiper_infer_status(<heading> <value1> [<value2> ...])
#   kuiper_infer_status(<heading> <condition> THEN <text for TRUE> ELSE <text for FALSE> )
function(kuiper_infer_status text)
    # 存储条件判断和对应的状态信息
    set(status_cond)
    set(status_then)
    set(status_else)

    set(status_current_name "cond")
    foreach(arg ${ARGN})
        if(arg STREQUAL "THEN")
            set(status_current_name "then")
        elseif(arg STREQUAL "ELSE")
            set(status_current_name "else")
        else()
            list(APPEND status_${status_current_name} ${arg})
        endif()
    endforeach()

    if(DEFINED status_cond)
        set(status_placeholder_length 23) # 设置占位符长度
        # 生成空字符串
        string(RANDOM LENGTH ${status_placeholder_length} ALPHABET " " status_placeholder)
        string(LENGTH "${text}" status_text_length) # 获取输入文本长度
        if(status_text_length LESS status_placeholder_length)
            # 截取字符串
            string(SUBSTRING "${text}${status_placeholder}" 0 ${status_placeholder_length} status_text)
        elseif(DEFINED status_then OR DEFINED status_else)
            message(STATUS "${text}")
            set(status_text "${status_placeholder}")
        else()
            set(status_text "${text}")
        endif()
        
        if(DEFINED status_then OR DEFINED status_else)
            if(${status_cond})
                # 将分号替换成空格,去除开头的空白字符
                string(REPLACE ";" " " status_then "${status_then}")
                string(REGEX REPLACE "^[ \t]+" "" status_then "${status_then}")
                message(STATUS "${status_text} ${status_then}")
            else()
                string(REPLACE ";" " " status_else "${status_else}")
                string(REGEX REPLACE "^[ \t]+" "" status_else "${status_else}")
                message(STATUS "${status_text} ${status_else}")
            endif()
        else()
            string(REPLACE ";" " " status_cond "${status_cond}")
            string(REGEX REPLACE "^[ \t]+" "" status_cond "${status_cond}")
            message(STATUS "${status_text} ${status_cond}")
        endif()
    else()
        message(STATUS "${text}")
    endif()
endfunction()

################################################################################################
# Function for fetching KuiperInfer version from git and headers
# Usage:
#   kuiper_infer_extract_git_version()
function(kuiper_infer_extract_git_version)
    set(KuiperInfer_GIT_VERSION "unkonw")
    find_package(Git)
    if(GIT_FOUND)
        execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
                        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
                        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                        OUTPUT_VARIABLE KuiperInfer_GIT_VERSION
                        RESULT_VARIABLE __git_result)         
        if(NOT ${__git_result} EQUAL 0)
            set(KuiperInfer_GIT_VERSION "unkonw")
        endif()     
    endif()

    set(KuiperInfer_GIT_VERSION ${KuiperInfer_GIT_VERSION} PARENT_SCOPE)
    set(KuiperInfer_VERSION "<TODO> (Caffe doesn't declare its version in headers)" PARENT_SCOPE)

    # caffe_parse_header(${Caffe_INCLUDE_DIR}/caffe/version.hpp Caffe_VERSION_LINES CAFFE_MAJOR CAFFE_MINOR CAFFE_PATCH)
    # set(Caffe_VERSION "${CAFFE_MAJOR}.${CAFFE_MINOR}.${CAFFE_PATCH}" PARENT_SCOPE)

    # or for #define Caffe_VERSION "x.x.x"
    # caffe_parse_header_single_define(Caffe ${Caffe_INCLUDE_DIR}/caffe/version.hpp Caffe_VERSION)
    # set(Caffe_VERSION ${Caffe_VERSION_STRING} PARENT_SCOPE)
endfunction()



################################################################################################
# Prints accumulated kuiper_infer configuration summary
# Usage:
#   kuiper_infer_print_configuration_summary()
function(kuiper_infer_print_configuration_summary)
    kuiper_infer_extract_git_version()

    kuiper_infer_merge_lists(__flags_rel CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS) 
    kuiper_infer_merge_lists(__flags_deb CMAKE_CXX_FLAGS_DEBUG   CMAKE_CXX_FLAGS)
    
    kuiper_infer_status("")
    kuiper_infer_status("******************* KuiperInfer Configuration Summary *******************")
    kuiper_infer_status("General:")
    kuiper_infer_status("  Version           :   ${KUIPERINFER_TARGET_VERSION}")     
    kuiper_infer_status("  Git               :   ${KuiperInfer_GIT_VERSION}")
    kuiper_infer_status("  System            :   ${CMAKE_SYSTEM_NAME}")
    kuiper_infer_status("  C++ compiler      :   ${CMAKE_CXX_COMPILER}")
    kuiper_infer_status("  Release CXX flags :   ${__flags_rel}")
    kuiper_infer_status("  Debug CXX flags   :   ${__flags_deb}")
    kuiper_infer_status("  Build type        :   ${CMAKE_BUILD_TYPE}")
    kuiper_infer_status("")
    kuiper_infer_status("Options:")
    kuiper_infer_status("  CPU_ONLY          :   ${CPU_ONLY}")
    kuiper_infer_status("  USE_CUDNN         :   ${USE_CUDNN}")
    kuiper_infer_status("  BUILD_DEMO        :   ${BUILD_DEMO}")
    kuiper_infer_status("  BUILD_BENCH       :   ${BUILD_BENCH}")
    kuiper_infer_status("  BUILD_TEST        :   ${BUILD_TEST}")
    kuiper_infer_status("")
    kuiper_infer_status("Dependencies:")
    kuiper_infer_status("  BLAS              : " BLAS_FOUND THEN "Yes" ELSE "No")
    kuiper_infer_status("  LAPACK            : " LAPACK_FOUND THEN "Yes" ELSE "No")
    kuiper_infer_status("  OpenMP            : " OpenMP_CXX_FOUND THEN "Yes (ver. ${OpenMP_CXX_VERSION})" ELSE "No")
    kuiper_infer_status("  glog              : " glog_FOUND THEN "Yes (ver. ${glog_VERSION})" ELSE "No")
    kuiper_infer_status("  Armadillo         : " ARMADILLO_FOUND THEN "Yes (ver. ${ARMADILLO_VERSION_STRING})" ELSE "No")
    
    if(HAVE_CUDA)
        kuiper_infer_status("")
        kuiper_infer_status("NVIDIA CUDA:")
        kuiper_infer_status("  Target GPU(s)     :   ${CUDA_ARCH_NAME}" )
        kuiper_infer_status("  GPU arch(s)       :   ${NVCC_FLAGS_EXTRA_readable}")
        if(USE_CUDNN)
            kuiper_infer_status("  cuDNN             : " HAVE_CUDNN THEN "Yes (ver. ${CUDNN_VERSION})" ELSE "Not found")
        else()
            kuiper_infer_status("  cuDNN             :   Disabled")
        endif()
    endif()
    
    if(BUILD_DOCS)
        kuiper_infer_status("")
        kuiper_infer_status("Documentaion:")
        kuiper_infer_status("  Doxygen           :" DOXYGEN_FOUND THEN "${DOXYGEN_EXECUTABLE} (${DOXYGEN_VERSION})" ELSE "No")
        kuiper_infer_status("  config_file       :   ${DOXYGEN_config_file}")
    endif()
    kuiper_infer_status("")
    kuiper_infer_status("Install:")
    kuiper_infer_status("  Install path      :   ${CMAKE_INSTALL_PREFIX}")

    kuiper_infer_status("*************************************************************************")
endfunction()


