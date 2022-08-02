get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
message(STATUS "MLIR_DIALECT_LIBS: ${dialect_libs}")
if(WITH_MLIR_CUDA_CODEGEN)
  set(MLIR_GPU_LIBS MLIRGPUToNVVMTransforms MLIRNVVMToLLVMIRTranslation)
endif(WITH_MLIR_CUDA_CODEGEN)

set(ONEFLOW_OP_GROUPS
    "ASSIGN;BINARY;BROADCAST;CONV;CROSS_ENTROPY;CUDA;DATASET;DETECTION;EAGER;FUSED;IDEMPOTENT;IDENTITY;IMAGE;INDICES;INVOLUTION;LOSS;MATH;MATMUL;MISC;NCCL;NORMALIZATION;OPTIMIZER;PADDING;PARALLEL_CAST;POOL;QUANTIZATION;REDUCE;RESHAPE;SCALAR;SOFTMAX;SUMMARY;TENSOR_BUFFER;TEST;TRIGONOMETRIC;UNARY;UPSAMPLE;ONE_EMBEDDING"
)

foreach(OP_GROUP_NAME IN LISTS ONEFLOW_OP_GROUPS)
  string(TOLOWER "${OP_GROUP_NAME}" OP_GROUP_NAME_LOWER)
  set(CPP_FILE "OneFlow.${OP_GROUP_NAME_LOWER}_ops.cpp")
  list(APPEND GROUPED_OP_CPP_FILES "${CPP_FILE}")
  configure_file(OneFlowOpGetGen.cpp.in ${CPP_FILE} @ONLY)
endforeach()

oneflow_add_mlir_dialect_library(
  MLIROneFlow
  SBP/SBPDialect.cpp
  SBP/SBPAttributes.cpp
  SBP/SBPImporter.cpp
  OneFlowDialect.cpp
  OneFlowTypes.cpp
  OneFlowOps.cpp
  OneFlowOpTraits.cpp
  OneFlowSupport.cpp
  OneFlowOpFolders.cpp
  Conversion/OneFlowToTosa.cpp
  Conversion/PTXToCubin.cpp
  Transform/BufferHostRegister.cpp
  Transform/OutlineAndFuse.cpp
  Transform/AutoNhwc.cpp
  Transform/ConvertInferenceOp.cpp
  Passes.cpp
  ${GROUPED_OP_CPP_FILES}
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/OneFlow
  DEPENDS
  MLIROneFlowOpsIncGen
  prepare_oneflow_third_party
  LINK_LIBS
  PUBLIC
  ${dialect_libs}
  MLIRTosaToLinalg
  MLIRMemRefToLLVM
  MLIRLinalgToLLVM
  MLIRSCFToGPU
  MLIRReconcileUnrealizedCasts
  ${MLIR_GPU_LIBS}
  MLIRIR
  oneflow)

if(WITH_MLIR_CUDA_CODEGEN)
  find_library(CUDA_DRIVER_LIBRARY cuda)
  target_link_libraries(MLIROneFlow PRIVATE ${CUDA_DRIVER_LIBRARY})
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif(WITH_MLIR_CUDA_CODEGEN)
