if(NOT BUILD_BENCH)
    return()
endif()

find_package(benchmark REQUIRED)

file(GLOB_RECURSE bench_srcs "${PROJECT_SOURCE_DIR}/bench/*.cpp")

# -----| adding bench target
add_executable(bench_test ${bench_srcs})

target_link_libraries(bench_test ${KuiperInfer_LINK} 
                    benchmark::benchmark benchmark::benchmark_main) 

kuiper_infer_default_properties(bench_test)
kuiper_infer_set_runtime_directory(bench_test "${PROJECT_BINARY_DIR}/bench")

# -----| install
install(TARGETS bench_test DESTINATION ${CMAKE_INSTALL_BINDIR})


if (MSVC)
    # find kuiper dll
    add_custom_command(TARGET bench_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:kuiper>/kuiper.dll"
            $<TARGET_FILE_DIR:bench_test>)
endif()

