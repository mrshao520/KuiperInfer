if(NOT BUILD_TEST)
    return()
endif()

find_package(GTest REQUIRED)

file(GLOB_RECURSE test_srcs "${PROJECT_SOURCE_DIR}/test/*.cpp")

# -----| adding test target
add_executable(kuiper_test ${test_srcs})
target_link_libraries(kuiper_test ${KuiperInfer_LINK} GTest::gtest)

kuiper_infer_default_properties(kuiper_test)

kuiper_infer_set_runtime_directory(kuiper_test "${PROJECT_BINARY_DIR}/test")
                    
install(TARGETS kuiper_test DESTINATION ${CMAKE_INSTALL_BINDIR})

# -----| adding runtest  please install C++ TestMate to see more details
add_test(NAME runtest COMMAND $<TARGET_FILE:kuiper_test>
                      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})          

if (MSVC)
    add_custom_command(TARGET kuiper_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:kuiper>/kuiper.dll"
            $<TARGET_FILE_DIR:kuiper_test>)
endif ()

