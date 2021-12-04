add_mlir_dialect_library(MLIROneFlowJITDialect
  OneFlowJITOps.cpp
  OneFlowJITDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/oneflow-jit/include

  DEPENDS
  MLIROneFlowJITOpsIncGen

  LINK_LIBS PUBLIC
  MLIRStandard
  MLIRIR
)
