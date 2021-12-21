set(LLVM_LINK_COMPONENTS
  Support
)
include(FetchContent)

FetchContent_Declare(
  inja
)
FetchContent_GetProperties(inja)
if(NOT inja_POPULATED)
  FetchContent_Populate(inja
    URL ${INJA_URL}
    URL_HASH MD5=${INJA_MD5}
  )
endif()
include_directories(${inja_SOURCE_DIR}/include/inja)

FetchContent_Declare(
  json
)
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json
    URL ${JSON_URL}
    URL_HASH MD5=${JSON_MD5}
  )
endif()
include_directories(${json_SOURCE_DIR}/include)

add_tablegen(oneflow_tblgen llvm
  tablegen.cpp
  op_schema_emitter.cpp
)

install(TARGETS oneflow_tblgen LLVMTableGen LLVMDemangle LLVMSupport COMPONENT OneFlowTableGen)
add_custom_target(install-oneflow-tblgen
  DEPENDS oneflow_tblgen
  COMMAND
      "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=OneFlowTableGen
      -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
)
