set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(oneflow_libs GLOBAL PROPERTY ALL_ONEFLOW_LIBS)

set(LLVM_ENABLE_RTTI ON) # turn this on to make it compatible with protobuf

add_llvm_executable(oneflow-gen-ods
  oneflow-gen-ods.cpp
  DEPENDS
  MLIROneFlow
  MLIROneFlowTranslation
)

set(_origin_prefix "\$ORIGIN")
if(APPLE)
  set(_origin_prefix "@loader_path")
endif()
set_target_properties(oneflow-gen-ods PROPERTIES
  BUILD_WITH_INSTALL_RPATH OFF
  BUILD_RPATH "${_origin_prefix}"
  INSTALL_RPATH "${_origin_prefix}"
)

llvm_update_compile_flags(oneflow-gen-ods)

target_link_libraries(oneflow-gen-ods
  PRIVATE
  ${dialect_libs}
  ${oneflow_libs}
)

mlir_check_link_libraries(oneflow-gen-ods)
