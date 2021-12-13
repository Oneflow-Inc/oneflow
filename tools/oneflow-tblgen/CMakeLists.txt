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
    URL https://github.com/pantor/inja/archive/refs/tags/v3.3.0.zip
    URL_HASH MD5=611e6b7206d0fb89728a3879f78b4775
  )
endif()
include_directories(${inja_SOURCE_DIR}/include/inja)

FetchContent_Declare(
  json
)
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json
    URL https://github.com/nlohmann/json/releases/download/v3.7.3/include.zip
    URL_HASH MD5=fb96f95cdf609143e998db401ca4f324
  )
endif()
include_directories(${json_SOURCE_DIR}/include)

add_tablegen(oneflow_tblgen llvm
  tablegen.cpp
  op_schema_emitter.cpp
)

add_custom_target(install-oneflow-tblgen
  DEPENDS oneflow_tblgen
  COMMAND
      "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=OneFlowTableGen
      -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
)
