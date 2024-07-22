# These lists are later turned into target properties on main kuiper_infer library target
set(KuiperInfer_LINKER_LIBS "")
set(KuiperInfer_INCLUDE_DIRS "")
set(KuiperInfer_DEFINITIONS "")
set(KuiperInfer_COMPILE_OPTIONS "")

# -----| phtread
if(!WIN32)
    list(APPEND KuiperInfer_LINKER_LIBS PUBLIC pthread)
endif()

# -----| OpenMP
find_package(OpenMP REQUIRED)
list(APPEND KuiperInfer_LINKER_LIBS PUBLIC OpenMP::OpenMP_CXX)
list(APPEND KuiperInfer_COMPILE_OPTIONS PUBLIC ${OpenMP_CXX_FLAGS})

# -----| glog
find_package(glog REQUIRED)
list(APPEND KuiperInfer_INCLUDE_DIRS PUBLIC ${glog_INCLUDE_DIRS})
list(APPEND KuiperInfer_LINKER_LIBS PUBLIC glog::glog)

# -----| Armadillo
find_package(Armadillo REQUIRED)
list(APPEND KuiperInfer_INCLUDE_DIRS PUBLIC ${ARMADILLO_INCLUDE_DIRS})
list(APPEND KuiperInfer_LINKER_LIBS PUBLIC ${ARMADILLO_LIBRARIES})

# -----| BLAS
find_package(BLAS REQUIRED)
list(APPEND KuiperInfer_LINKER_LIBS PUBLIC ${BLAS_LIBRARIES})

# -----| LAPACK
find_package(LAPACK REQUIRED)
list(APPEND KuiperInfer_LINKER_LIBS PUBLIC ${LAPACK_LIBRARIES})


