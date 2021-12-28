set(LLVM_LINK_COMPONENTS
  Support
)
include(FetchContent)

set(JSON_Install ON)
FetchContent_Declare(
  json
  URL ${JSON_URL}
  URL_HASH MD5=${JSON_URL_HASH}
)

set(INJA_USE_EMBEDDED_JSON OFF)
set(INJA_BUILD_TESTS OFF)
set(BUILD_BENCHMARK OFF)
FetchContent_Declare(
  inja
  URL ${INJA_URL}
  URL_HASH MD5=${INJA_URL_HASH}
)

FetchContent_MakeAvailable(json inja)

add_tablegen(oneflow_tblgen llvm
  tablegen.cpp
  op_schema_emitter.cpp
)

target_link_libraries(obj.oneflow_tblgen PRIVATE
  nlohmann_json::nlohmann_json
  pantor::inja
)

install(TARGETS oneflow_tblgen LLVMTableGen LLVMDemangle LLVMSupport COMPONENT OneFlowTableGen)
add_custom_target(install-oneflow-tblgen
  DEPENDS oneflow_tblgen
  COMMAND
      "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=OneFlowTableGen
      -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
)
