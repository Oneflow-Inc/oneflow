get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
message(STATUS "MLIR_DIALECT_LIBS: ${dialect_libs}")
if(WITH_MLIR_CUDA_CODEGEN)
set(MLIR_GPU_LIBS MLIRSCFToGPU MLIRGPUToNVVMTransforms MLIRNVVMToLLVMIRTranslation)
endif(WITH_MLIR_CUDA_CODEGEN)

add_mlir_dialect_library(MLIROneFlow
  OneFlowDialect.cpp
  OneFlowOps.cpp
  Conversion/OneFlowToTosa.cpp
  Conversion/SCFToGPU.cpp
  Conversion/PTXToCubin.cpp
  Transform/BufferHostRegister.cpp
  Transform/OutlineAndFuse.cpp
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/OneFlow

  DEPENDS
  MLIROneFlowOpsIncGen

  LINK_LIBS PUBLIC
  ${dialect_libs}
  MLIRTosaToLinalg
  MLIRSCFToStandard
  MLIRMemRefToLLVM
  MLIRLinalgToLLVM
  ${MLIR_GPU_LIBS}
  MLIRIR
)

if(WITH_MLIR_CUDA_CODEGEN)
  find_library(CUDA_DRIVER_LIBRARY cuda)
  target_link_libraries(MLIROneFlow
    PRIVATE
    ${CUDA_DRIVER_LIBRARY}
  )
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif(WITH_MLIR_CUDA_CODEGEN)
