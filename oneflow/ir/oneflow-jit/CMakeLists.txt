set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_ENABLE_RTTI ON) # turn this on to make it compatible with protobuf

include_directories(${PROJECT_SOURCE_DIR}/oneflow-jit/include)
include_directories(${PROJECT_BINARY_DIR}/oneflow-jit/include)

add_subdirectory(include)
add_subdirectory(lib)
