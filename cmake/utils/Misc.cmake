# -----| configuration types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Possible configurations" FORCE)
mark_as_advanced(CMAKE_CONFIGURATION_TYPES)

if(DEFINED CMAKE_BUILD_TYPE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()

# -----| If user done't specify build type then assume release
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_COMPILER_IS_CLANGXX TRUE)
endif()

# -----| Install options
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/install" CACHE PATH "Default install path" FORCE)
endif()

# -----| RPATH settings
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Use link paths for shared library rpath")
set(CMAKE_MACOSX_RPATH TRUE)

list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
     ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} __is_systtem_dir)
if(${__is_systtem_dir} STREQUAL -1)
  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
endif()

# -----| Set debug postfix
set(KuiperInfer_DEBUG_POSTFIX "-d")
set(KuiperInfer_POSTFIX "")
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(KuiperInfer_POSTFIX ${KuiperInfer_DEBUG_POSTFIX})
endif()