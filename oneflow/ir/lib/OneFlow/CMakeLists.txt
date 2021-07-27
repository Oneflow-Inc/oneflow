get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
message(STATUS "MLIR_DIALECT_LIBS: ${dialect_libs}")
add_mlir_dialect_library(MLIROneFlow
  OneFlowDialect.cpp
  OneFlowOps.cpp
  Conversion/OneFlowToTosa.cpp
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
  MLIRIR
)
