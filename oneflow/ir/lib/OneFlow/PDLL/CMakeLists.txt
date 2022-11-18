add_mlir_pdll_library(MLIRTestPDLLPatternsIncGen
  TestPDLL.pdll
  TestPDLLPatterns.h.inc
  )
add_mlir_library(MLIRTestPDLL
  TestPDLL.cpp

  DEPENDS
  MLIRTestPDLLPatternsIncGen
  )

