include (ExternalProject)

set(EIGEN_INCLUDE_DIR ${THIRD_PARTY_DIR}/eigen/include/eigen3)
set(EIGEN_INSTALL_DIR ${THIRD_PARTY_DIR}/eigen)

if(WITH_XLA)
  #set(EIGEN_URL "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/386d809bde475c65b7940f290efe80e6a05878c4/eigen-386d809bde475c65b7940f290efe80e6a05878c4.tar.gz")
  set(EIGEN_URL "https://gitlab.com/libeigen/eigen/-/archive/386d809bde475c65b7940f290efe80e6a05878c4/eigen-386d809bde475c65b7940f290efe80e6a05878c4.tar.gz")
  set(EIGEN_MD5 67b12e85555e0ac97b4cf8bae7fd65ad)
  else()
  set(EIGEN_URL https://github.com/Oneflow-Inc/eigen-git-mirror/archive/e9e95489a.tar.gz)
  set(EIGEN_MD5 a23cb70e12d1bf9b09cb28af51bc26ae)
endif()
use_mirror(VARIABLE EIGEN_URL URL ${EIGEN_URL})

add_definitions(-DEIGEN_NO_AUTOMATIC_RESIZING -DEIGEN_USE_GPU)
if (NOT WITH_XLA)
add_definitions(-DEIGEN_NO_MALLOC)
endif()
#add_definitions(-DEIGEN_NO_AUTOMATIC_RESIZING -DEIGEN_NO_MALLOC -DEIGEN_USE_GPU)

if (THIRD_PARTY)

ExternalProject_Add(eigen
    PREFIX eigen
    URL ${EIGEN_URL}
    URL_MD5 ${EIGEN_MD5}
    UPDATE_COMMAND ""
    INSTALL_DIR "${EIGEN_INSTALL_DIR}"
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DCMAKE_INSTALL_PREFIX:STRING=${EIGEN_INSTALL_DIR}
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
        -DBUILD_TESTING:BOOL=OFF
)


endif(THIRD_PARTY)
