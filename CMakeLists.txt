# -----| cmake version
cmake_minimum_required(VERSION 3.16)

# -----| kuiper_infer project
project(kuiper_infer LANGUAGES C CXX)

# -----| cpp version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----| kuiper_infer version
set(KUIPERINFER_TARGET_VERSION "0.1.1" CACHE STRING "KuiperInfer logical version" FORCE)
set(KUIPERINFER_TARGET_SOVERSION "0.1.1" CACHE STRING "KuiperInfer soname version" FORCE)

# -----| using cmake scripts and modules
list(APPEND  CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(GNUInstallDirs)
include(cmake/utils/Utils.cmake)
include(cmake/utils/Misc.cmake)
include(cmake/utils/Summary.cmake)
include(cmake/utils/ConfigGen.cmake)

# -----| options 
kuiper_infer_option(CPU_ONLY "Build kuiper_infer without CUDA support" ON)
kuiper_infer_option(USE_CUDNN "Build kuiper_infer with cuDNN library support" ON IF NOT CPU_ONLY)
kuiper_infer_option(BUILD_DEMO "Build the demo project" ON)
kuiper_infer_option(BUILD_BENCH "Build the bench project" ON)
kuiper_infer_option(BUILD_TEST "Build the test project" ON)

# -----| dependencies
include(cmake/utils/Dependencies.cmake)

# -----| flags
if(UNIX OR APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    # 假如编译x64，那么SSE2永远可用
    # 假如编译x86，那么SSE2可用的话，就开启SSE2
    if (CMAKE_SIZEOF_VOID_P EQUAL 4)
        # check if SSE2 is available
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("/arch:SSE2" HAS_SSE2)
    else ()
        set(HAS_SSE2 ON)
    endif ()
    if (HAS_SSE2)
        # set __SSE2__ and __XOP__ macros
        add_definitions(-D__SSE2__ -D__XOP__)
    endif ()
    # Force LLVM OpenMP on MSVC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm /openmp:experimental /arch:AVX2")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

set_kuiper_infer_link()

# -----| warning
kuiper_infer_warnings_disable(CMAKE_CXX_FLAGS -Wno-uninitialized -Wno-reorder)

# -----| definitions
add_definitions(-DKUIPERINFER_VERSION=${KUIPERINFER_TARGET_VERSION})
add_compile_definitions(SSE_MATHFUN_WITH_CODE USE_SSE_AUTO)

# -----| includes
set(KuiperInfer_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(KuiperInfer_SRC_DIR ${PROJECT_SOURCE_DIR}/source)
include_directories(${PROJECT_BINARY_DIR})

# -----| enable testing
if (BUILD_TEST)
    enable_testing()
endif()

# -----| subdirectories
add_subdirectory(source)
add_subdirectory(demos)
add_subdirectory(bench)
add_subdirectory(test)

# -----| uninstall target
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/Uninstall.cmake.in 
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake 
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P 
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake)

# -----| configuration summary
kuiper_infer_print_configuration_summary()

# -----| export configs generation
kuiper_infer_generate_export_configs()