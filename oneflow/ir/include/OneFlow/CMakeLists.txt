# set(ONEFLOW_USER_OP_GEN_TD_PATH "${PROJECT_BINARY_DIR}/include/OneFlow")
set(ONEFLOW_USER_OP_GEN_TD_PATH "${PROJECT_SOURCE_DIR}/include/OneFlow")

set(LLVM_TARGET_DEFINITIONS OneFlowEnums.td)
mlir_tablegen(OneFlowEnums.h.inc -gen-enum-decls)
mlir_tablegen(OneFlowEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIROneFlowEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS OneFlowPatterns.td)
set(ONEFLOW_OP_GROUPS_USED_IN_PATTERNS
    "SCALAR;UNARY;FUSED;MISC;BINARY;IDEMPOTENT;NORMALIZATION;MATMUL;BROADCAST;CONV;PADDING")
foreach(OP_GROUP_NAME IN LISTS ONEFLOW_OP_GROUPS_USED_IN_PATTERNS)
  list(APPEND LLVM_TABLEGEN_FLAGS "-DGET_ONEFLOW_${OP_GROUP_NAME}_OP_DEFINITIONS")
endforeach()
mlir_tablegen(OneFlowPatterns.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIROneFlowPatternsIncGen)

# NOTE: seperate conversion and opt with --name
set(LLVM_TARGET_DEFINITIONS OneFlowPasses.td)
mlir_tablegen(OneFlowPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIROneFlowPassIncGen)

set(LLVM_TABLEGEN_FLAGS "")
add_mlir_interface(OneFlowInterfaces)

set(LLVM_TARGET_DEFINITIONS OneFlowOpGetGen.td)

set(ONEFLOW_OP_GROUPS
    "ASSIGN;BINARY;BROADCAST;CONV;CROSS_ENTROPY;CUDA;DATASET;DETECTION;EAGER;FUSED;IDEMPOTENT;IDENTITY;IMAGE;INDICES;INVOLUTION;LOSS;MATH;MATMUL;MISC;NCCL;NORMALIZATION;OPTIMIZER;PADDING;PARALLEL_CAST;POOL;QUANTIZATION;REDUCE;RESHAPE;SCALAR;SOFTMAX;SUMMARY;TENSOR_BUFFER;TEST;TRIGONOMETRIC;UNARY;UPSAMPLE;ONE_EMBEDDING;LINEAR_ALGEBRA;SYSTEM;MLIR_JIT"
)
foreach(OP_GROUP_NAME IN LISTS ONEFLOW_OP_GROUPS)
  message(STATUS "Enable OneFlow MLIR op group: ${OP_GROUP_NAME}")
  set(ONE_LLVM_TABLEGEN_FLAGS "-DGET_ONEFLOW_${OP_GROUP_NAME}_OP_DEFINITIONS")
  list(APPEND FULL_LLVM_TABLEGEN_FLAGS "${ONE_LLVM_TABLEGEN_FLAGS}")
  set(LLVM_TABLEGEN_FLAGS "${ONE_LLVM_TABLEGEN_FLAGS}")
  string(TOLOWER "${OP_GROUP_NAME}" OP_GROUP_NAME_LOWER)
  set(CPP_INC_FILE "OneFlow.${OP_GROUP_NAME_LOWER}_ops.cpp.inc")
  set(HEADER_INC_FILE "OneFlow.${OP_GROUP_NAME_LOWER}_ops.h.inc")
  mlir_tablegen(${CPP_INC_FILE} -gen-op-defs)
  mlir_tablegen(${HEADER_INC_FILE} -gen-op-decls)
endforeach()
add_public_tablegen_target(MLIROneFlowOpGroupDefsIncGen)

set(LLVM_TABLEGEN_FLAGS "${FULL_LLVM_TABLEGEN_FLAGS}")
mlir_tablegen(OneFlow.gen_ops.h.inc -gen-op-decls)
add_public_tablegen_target(MLIROneFlowOpGroupDeclsIncGen)

set(LLVM_TARGET_DEFINITIONS SBP/SBPOps.td)
mlir_tablegen(SBPDialect.h.inc -gen-dialect-decls)
mlir_tablegen(SBPDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(SBPAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SBPAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRSBPIncGen)

set(LLVM_TARGET_DEFINITIONS OKL/OKLOps.td)
mlir_tablegen(OKLDialect.h.inc -gen-dialect-decls -dialect=okl)
mlir_tablegen(OKLDialect.cpp.inc -gen-dialect-defs -dialect=okl)
mlir_tablegen(OKLOps.h.inc -gen-op-decls)
mlir_tablegen(OKLOps.cpp.inc -gen-op-defs)
mlir_tablegen(OKLTypes.h.inc -gen-typedef-decls)
mlir_tablegen(OKLTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(OKLPasses.h.inc -gen-pass-decls)
mlir_tablegen(OKLEnums.h.inc -gen-enum-decls)
mlir_tablegen(OKLEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(OKLAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(OKLAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIROKLIncGen)

set(LLVM_TARGET_DEFINITIONS OKM/OKMOps.td)
mlir_tablegen(OKMDialect.h.inc -gen-dialect-decls -dialect=okm)
mlir_tablegen(OKMDialect.cpp.inc -gen-dialect-defs -dialect=okm)
mlir_tablegen(OKMOps.h.inc -gen-op-decls)
mlir_tablegen(OKMOps.cpp.inc -gen-op-defs)
mlir_tablegen(OKMPasses.h.inc -gen-pass-decls)
mlir_tablegen(OKMAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(OKMAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIROKMIncGen)

set(LLVM_TABLEGEN_FLAGS "")
add_mlir_dialect(
  OneFlowOps
  oneflow
  DEPENDS
  MLIRSBPIncGen
  MLIROneFlowEnumsIncGen
  MLIROneFlowPatternsIncGen
  MLIROneFlowPassIncGen
  MLIROneFlowInterfacesIncGen
  MLIROneFlowOpGroupDefsIncGen
  MLIROneFlowOpGroupDeclsIncGen)
