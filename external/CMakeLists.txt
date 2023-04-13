set(ONETBB_URL
    https://github.com/oneapi-src/oneTBB/archive/3db67b5ba2a81bd1288325c5847e09e13c46f4d7.zip)
use_mirror(VARIABLE ONETBB_URL URL ${ONETBB_URL})
set(ONETBB_MD5 7545d4084baff17af73da2dae5ab8005)

set(ROBIN_HOOD_HASHING_URL
    https://github.com/martinus/robin-hood-hashing/archive/refs/tags/3.11.5.tar.gz)
use_mirror(VARIABLE ROBIN_HOOD_HASHING_URL URL ${ROBIN_HOOD_HASHING_URL})
set(ROBIN_HOOD_HASHING_MD5 a78bd30a7582f25984f8592652836467)

set(FMT_URL https://github.com/fmtlib/fmt/archive/fc07217d85e6dcec52878807d6bbd89a9d9156a5.zip)
use_mirror(VARIABLE FMT_URL URL ${FMT_URL})
set(FMT_MD5 7d9bb2ececc9ede29cd35bdc42a7e22c)

set(KINETO_URL
    https://github.com/pytorch/kineto/archive/ff8dba20499a660650632952be76450bd70a52a6.zip)
use_mirror(VARIABLE KINETO_URL URL ${KINETO_URL})
set(KINETO_MD5 f9b550591b3899fb267270c19484933f)

set(EXTERNAL_TARGETS)

if(WITH_TBB) # set(WITH_${threading_runtime_item} ON) in threading.cmake
  add_subdirectory(onetbb)
  list(APPEND EXTERNAL_TARGETS tbb)
endif()

add_subdirectory(robin-hood-hashing)
list(APPEND EXTERNAL_TARGETS robin_hood)

add_subdirectory(fmt)
list(APPEND EXTERNAL_TARGETS fmt)

add_subdirectory(kineto)
list(APPEND EXTERNAL_TARGETS kineto)

mark_targets_as_system(${EXTERNAL_TARGETS})

set_property(GLOBAL PROPERTY EXTERNAL_TARGETS ${EXTERNAL_TARGETS})
