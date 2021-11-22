configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(LLVM_EXTERNAL_LIT "${llvm_monorepo_BINARY_DIR}/bin/llvm-lit" CACHE STRING "")
message(STATUS "LLVM_EXTERNAL_LIT: ${LLVM_EXTERNAL_LIT}")

set(ONEFLOW_TEST_DEPENDS
        FileCheck count not
        oneflow-opt
        oneflow-translate
        )

add_lit_testsuite(check-oneflow "Running the OneFlow MLIR regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${ONEFLOW_TEST_DEPENDS}
        )
set_target_properties(check-oneflow PROPERTIES FOLDER "Tests")

add_lit_testsuites(ONEFLOW ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${ONEFLOW_TEST_DEPENDS})

add_custom_target(c1 DEPENDS check-oneflow)
