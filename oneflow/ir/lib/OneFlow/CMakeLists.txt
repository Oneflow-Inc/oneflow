# TODO: linking dialect_libs only for the return op
# FIXME: remove me when outline jit func is defined and saved in a separated module
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_mlir_dialect_library(MLIROneFlow
  OneFlowDialect.cpp
  OneFlowOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/OneFlow

  DEPENDS
  MLIROneFlowOpsIncGen

  LINK_LIBS PUBLIC
  ${dialect_libs}
  MLIRIR
)
