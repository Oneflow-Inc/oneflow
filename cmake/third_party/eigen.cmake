include (ExternalProject)

set(EIGEN_INCLUDE_DIR ${THIRD_PARTY_DIR}/eigen/include/eigen3)
set(EIGEN_INSTALL_DIR ${THIRD_PARTY_DIR}/eigen)

# set(eigen_URL https://github.com/eigenteam/eigen-git-mirror)
# set(eigen_TAG e9e95489a0b241412e31f0525e85b2fab386c786)

set(eigen_URL http://download.oneflow.org/eigenteam-eigen-git-mirror-3.3.0-690-ge9e9548.tar.gz)

add_definitions(-DEIGEN_NO_AUTOMATIC_RESIZING -DEIGEN_NO_MALLOC -DEIGEN_USE_GPU)

if (BUILD_THIRD_PARTY)
  
ExternalProject_Add(eigen
    PREFIX eigen
    # GIT_REPOSITORY ${eigen_URL}
    # GIT_TAG ${eigen_TAG}
    URL ${eigen_URL}
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


endif(BUILD_THIRD_PARTY)
