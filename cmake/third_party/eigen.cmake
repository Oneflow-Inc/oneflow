include(ExternalProject)

set(EIGEN_INCLUDE_DIR ${THIRD_PARTY_DIR}/eigen/include/eigen3)
set(EIGEN_INSTALL_DIR ${THIRD_PARTY_DIR}/eigen)

set(EIGEN_URL https://github.com/Oneflow-Inc/eigen-git-mirror/archive/refs/tags/e9e95489a.tar.gz)
set(EIGEN_MD5 a23cb70e12d1bf9b09cb28af51bc26ae)
use_mirror(VARIABLE EIGEN_URL URL ${EIGEN_URL})

if(BUILD_CUDA)
  add_definitions(-DEIGEN_USE_GPU)
endif()

if(THIRD_PARTY)

  # This pinned Eigen dev snapshot has a latent bug in Transpositions.h: the friend
  # operator* of Transpose<TranspositionsBase<...>> calls trt.derived(), but that class
  # has no derived() member (it is a standalone expression, not a CRTP base). g++ never
  # instantiates the offending friend so it slips through; clang's stricter two-phase
  # lookup rejects it. The monolith is compiled with the ROCm clang, so patch the call to
  # pass the Transpose object directly (matches the sibling operator* on the next lines,
  # which already pass *this). Not HIP-specific; needed for any clang build.
  set(EIGEN_TRANSPOSITIONS_H Eigen/src/Core/Transpositions.h)
  ExternalProject_Add(
    eigen
    PREFIX eigen
    URL ${EIGEN_URL}
    URL_MD5 ${EIGEN_MD5}
    PATCH_COMMAND sed -i "s/matrix.derived(), trt.derived()/matrix.derived(), trt/"
                  <SOURCE_DIR>/${EIGEN_TRANSPOSITIONS_H}
    UPDATE_COMMAND ""
    INSTALL_DIR "${EIGEN_INSTALL_DIR}"
    CMAKE_CACHE_ARGS
      -DCMAKE_C_COMPILER_LAUNCHER:STRING=${CMAKE_C_COMPILER_LAUNCHER}
      -DCMAKE_CXX_COMPILER_LAUNCHER:STRING=${CMAKE_CXX_COMPILER_LAUNCHER}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
      -DCMAKE_INSTALL_PREFIX:STRING=${EIGEN_INSTALL_DIR}
      -DCMAKE_INSTALL_MESSAGE:STRING=${CMAKE_INSTALL_MESSAGE}
      -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
      -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
      -DBUILD_TESTING:BOOL=OFF)

endif(THIRD_PARTY)
