get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIROneFlow
        )
add_llvm_executable(oneflow-opt oneflow-opt.cpp)

llvm_update_compile_flags(oneflow-opt)
target_link_libraries(oneflow-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(oneflow-opt)
