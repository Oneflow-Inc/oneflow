add_subdirectory(Dialect)

add_mlir_library(MLIROneFlowJIT
  JITImporter.cpp
  JITinterpreter.cpp
  JITModule.cpp
  EXCLUDE_FROM_LIBMLIR
  ADDITIONAL_HEADERS
  ${PROJECT_SOURCE_DIR}/oneflow-translate/include
  ${PROJECT_SOURCE_DIR}/oneflow-jit/include
  DEPENDS oneflow_deps
	LINK_LIBS PUBLIC
	MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRSupport
  MLIROneFlow
  MLIRExecutionEngine
  MLIROneFlowTranslation
  MLIROneFlowJITDialect
  ENABLE_AGGREGATION
)
target_include_directories(MLIROneFlowJIT PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
