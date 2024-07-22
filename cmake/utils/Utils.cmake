################################################################################################
# Clears variables from list
# Usage:
#   kuiper_infer_clear_vars(<variables_list>)
macro(kuiper_infer_clear_vars)
    foreach(_var ${ARGN})
        unset(${_var})
    endforeach()
endmacro()


########################################################################################################
# An option that the user can select. Can accept condition to control when option is available for user.
# 用户可以选择的选项。可以接受条件来控制选项何时可供用户使用。
# Usage:
#   kuiper_infer_option(<option_variable> "doc string" <initial value or boolean expression> [IF <condition>])
#   kuiper_infer_option(变量名 描述 默认值)
function(kuiper_infer_option variable description value)
    set(__value ${value}) # 存储默认值
    set(__condition "")   # 存储条件变量
    set(__varname "__value")
    foreach(arg ${ARGN}) # 遍历所有额外参数 IF;TEST
        if(arg STREQUAL "IF" OR arg STREQUAL "if")
            set(__varname "__condition")
        else()
            list(APPEND ${__varname} ${arg}) # ${${__varname}} 
        endif()
    endforeach()
    
    unset(__varname) # 清空__varname
    if("${__condition}" STREQUAL "")
        set(__condition 2 GREATER 1) # 设置总为真的条件
    endif()

    if(${__condition})
        if("${__value}" MATCHES ";")
            if(${__value})
                option(${variable} "${description}" ON)
            else()
                option(${variable} "${description}" OFF)
            endif()
        elseif(DEFINED ${__value})
            if(${__value})
                option(${variable} "${description}" ON)
            else()
                option(${variable} "${description}" OFF)
            endif()
        else()
            option(${variable} "${description}" ${__value})
        endif()
    else()
        unset(${variable} CACHE)
    endif()
endfunction()

################################################################################################
# Function merging lists to single string.
# Usage:
#   kuiper_infer_merge_lists(out_variable <list1> [<list2>] [<list3>] ...)
function(kuiper_infer_merge_lists out_var)
    set(__result "")
    foreach(__list ${ARGN})
        foreach(__list_sub ${${__list}})
            # 去除 __list_sub 前后空白符
            string(STRIP ${__list_sub} __list_sub)
            set(__result "${__result} ${__list_sub}")
        endforeach() 
    endforeach()
    string(STRIP ${__result} __result)
    # 使用 PARENT_SCOPE 确保变量在函数外部可见
    set(${out_var} ${__result} PARENT_SCOPE)
endfunction()


################################################################################################
# Command for disabling warnings for different platforms (see below for gcc and VisualStudio)
# 将 warning 放入 CMAKE_C_FLAGS 或者 CMAKE_CXX_FLAGS
# Usage:
#   kuiper_infer_warnings_disable(<CMAKE_[C|CXX]_FLAGS[_CONFIGURATION]> -Wshadow /wd4996 ..,)
# Example:
#   kuiper_infer_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)
#       before CMAKE_CXX_FLAGS :  -fPIC -Wall
#       after  CMAKE_CXX_FLAGS :  -fPIC -Wall -Wno-sign-compare -Wno-uninitialized
macro(kuiper_infer_warnings_disable)
    set(_flag_vars "")
    set(_msvc_warnings "")
    set(_gxx_warnings "")

    foreach(arg ${ARGN})
        if(arg MATCHES "^CMAKE_")
            list(APPEND _flag_vars ${arg})
        elseif(arg MATCHES "^/wd")
            list(APPEND _msvc_warnings ${arg})
        elseif(arg MATCHES "^-W")
            list(APPEND _gxx_warnings ${arg})
        endif()
    endforeach()

    if(NOT _flag_vars)
        set(_flag_vars CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    endif()

    if(MSVC AND _msvc_warnings)
        foreach(var ${_flag_vars})
            foreach(warning ${_msvc_warnings})
                set(${var} "${${var}} ${warning}")
            endforeach()
        endforeach()
    elseif((CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX) AND _gxx_warnings)
        foreach(var ${_flag_vars})
            foreach(warning ${_gxx_warnings})
                if(NOT warning MATCHES "^-Wno-")
                    string(REPLACE "${warning}" "" ${var} "${${var}}")
                    string(REPLACE "-W" "-Wno-" warning "${warning}")
                endif()
                set(${var} "${${var}} ${warning}")
            endforeach()
        endforeach()
    endif()

    kuiper_infer_clear_vars(_flag_vars _msvc_warnings _gxx_warnings)
endmacro()


################################################################################################
# Defines global KuiperInfer_LINK flag, This flag is required to prevent linker from excluding
# some objects which are not addressed directly but are registered via static constructors
macro(set_kuiper_infer_link)
    if(BUILD_SHARED_LIBS)
        set(KuiperInfer_LINK kuiper)
    else()
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            set(KuiperInfer_LINK -Wl,-force_load kuiper)
        elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            set(KuiperInfer_LINK -Wl,--whole-archive kuiper -Wl,--no-whole-archive)
        endif()
    endif()
endmacro()


################################################################################################
# Short command for setting default target properties
# Usage:
#   kuiper_infer_default_properties(<target>)
function(kuiper_infer_default_properties target)
    set_target_properties(${target} PROPERTIES
            DEBUG_POSTFIX ${KuiperInfer_DEBUG_POSTFIX}
            ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
            LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
endfunction()

################################################################################################
# Short command for setting solution folder property for target
# Usage:
#   kuiper_infer_set_solution_folder(<target> <folder>)
function(kuiper_infer_set_solution_folder target folder)
  if(USE_PROJECT_FOLDERS)
    set_target_properties(${target} PROPERTIES FOLDER "${folder}")
  endif()
endfunction()

################################################################################################
# Short command for setting runtime directory for build target
# 生成RUNTIME目标文件的输出目录
# Usage:
#   kuiper_infer_set_runtime_directory(<target> <dir>)
function(kuiper_infer_set_runtime_directory target dir)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${dir}")
endfunction()



