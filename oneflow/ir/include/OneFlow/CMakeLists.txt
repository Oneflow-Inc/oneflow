set(ONEFLOW_USER_OP_GEN_TD_PATH "${PROJECT_BINARY_DIR}/include/OneFlow")
message(STATUS "Generating user op ODS ${ONEFLOW_USER_OP_GEN_TD_PATH}/OneFlowUserOpGen.td")
add_custom_target(GenUserOpODS
  DEPENDS oneflow-gen-ods
  COMMAND "$<TARGET_FILE:oneflow-gen-ods>"
  BYPRODUCTS OneFlowUserOpGen.td
  WORKING_DIRECTORY "${ONEFLOW_USER_OP_GEN_TD_PATH}"
)

set(LLVM_TARGET_DEFINITIONS OneFlowEnums.td)
mlir_tablegen(OneFlowEnums.h.inc -gen-enum-decls)
mlir_tablegen(OneFlowEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIROneFlowEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS OneFlowPatterns.td)
set(ONEFLOW_OP_GROUPS_USED_IN_PATTERNS "SCALAR;UNARY;FUSED;MISC;BINARY;IDEMPOTENT;NORMALIZATION")
foreach (OP_GROUP_NAME IN LISTS ONEFLOW_OP_GROUPS_USED_IN_PATTERNS)
    list(APPEND LLVM_TABLEGEN_FLAGS "-DGET_ONEFLOW_${OP_GROUP_NAME}_OP_DEFINITIONS")
endforeach()
mlir_tablegen(OneFlowPatterns.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIROneFlowPatternsIncGen)
add_dependencies(MLIROneFlowPatternsIncGen GenUserOpODS)

# NOTE: seperate conversion and opt with --name
set(LLVM_TARGET_DEFINITIONS OneFlowOps.td)
mlir_tablegen(OneFlowPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIROneFlowPassIncGen)

set(LLVM_TABLEGEN_FLAGS "")
add_mlir_interface(OneFlowInterfaces)

set(LLVM_TARGET_DEFINITIONS OneFlowOpGetGen.td)

set(ONEFLOW_OP_GROUPS "ASSIGN;BINARY;BROADCAST;CONV;CROSS_ENTROPY;CUDA;DATASET;DETECTION;EAGER;FUSED;IDEMPOTENT;IDENTITY;IMAGE;INDICES;INVOLUTION;LOSS;MATH;MATMUL;MISC;NCCL;NORMALIZATION;OPTIMIZER;PADDING;PARALLEL_CAST;POOL;QUANTIZATION;REDUCE;RESHAPE;SCALAR;SOFTMAX;SUMMARY;TENSOR_BUFFER;TEST;TRIGONOMETRIC;UNARY;UPSAMPLE")
foreach (OP_GROUP_NAME IN LISTS ONEFLOW_OP_GROUPS)
    message(STATUS "Enable OneFlow MLIR op group: ${OP_GROUP_NAME}")
    set(ONE_LLVM_TABLEGEN_FLAGS "-DGET_ONEFLOW_${OP_GROUP_NAME}_OP_DEFINITIONS")
    list(APPEND FULL_LLVM_TABLEGEN_FLAGS "${ONE_LLVM_TABLEGEN_FLAGS}")
    set(LLVM_TABLEGEN_FLAGS "${ONE_LLVM_TABLEGEN_FLAGS}")
    string(TOLOWER "${OP_GROUP_NAME}" OP_GROUP_NAME_LOWER)
    set(CPP_INC_FILE "OneFlow.${OP_GROUP_NAME_LOWER}_ops.cpp.inc")
    mlir_tablegen(${CPP_INC_FILE} -gen-op-defs)
endforeach()
add_public_tablegen_target(MLIROneFlowOpGroupDefsIncGen)
add_dependencies(MLIROneFlowOpGroupDefsIncGen GenUserOpODS)

set(LLVM_TABLEGEN_FLAGS "${FULL_LLVM_TABLEGEN_FLAGS}")
mlir_tablegen(OneFlow.Ops.h.inc -gen-op-decls)
add_public_tablegen_target(MLIROneFlowOpGroupDeclsIncGen)
add_dependencies(MLIROneFlowOpGroupDeclsIncGen GenUserOpODS)

set(LLVM_TABLEGEN_FLAGS "")
add_mlir_dialect(
    OneFlowOps oneflow DEPENDS
        MLIROneFlowEnumsIncGen
        MLIROneFlowPatternsIncGen
        MLIROneFlowPassIncGen
        MLIROneFlowInterfacesIncGen
        MLIROneFlowOpGroupDefsIncGen
        MLIROneFlowOpGroupDeclsIncGen
)
