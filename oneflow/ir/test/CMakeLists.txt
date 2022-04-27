llvm_canonicalize_cmake_booleans(WITH_MLIR_CUDA_CODEGEN BUILD_CUDA)

message(STATUS "LLVM_TOOLS_BINARY_DIR (used as LLVM_TOOLS_DIR): ${LLVM_TOOLS_BINARY_DIR}")
message(STATUS "LLVM_EXTERNAL_LIT: ${LLVM_EXTERNAL_LIT}")
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(ONEFLOW_TEST_DEPENDS FileCheck count not oneflow-opt oneflow-translate)

add_lit_testsuite(
  check-oneflow "Running the OneFlow MLIR regression tests from: ${CMAKE_CURRENT_SOURCE_DIR}"
  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${ONEFLOW_TEST_DEPENDS})
set_target_properties(check-oneflow PROPERTIES FOLDER "Tests")
if(LLVM_PROVIDER STREQUAL "in-tree")
  add_dependencies(check-oneflow mlir-cpu-runner)
endif()
add_dependencies(check-oneflow oneflow_internal)
add_dependencies(check-oneflow oneflow-runner)
add_lit_testsuites(ONEFLOW ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${ONEFLOW_TEST_DEPENDS})

add_custom_target(c1 DEPENDS check-oneflow)
