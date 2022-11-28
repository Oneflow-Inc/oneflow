set(LLVM_LINK_COMPONENTS Support)
include(FetchContent)

set(JSON_Install ON CACHE STRING "" FORCE)
FetchContent_Declare(json URL ${JSON_URL} URL_HASH MD5=${JSON_URL_HASH})

set(INJA_USE_EMBEDDED_JSON OFF CACHE STRING "" FORCE)
set(INJA_BUILD_TESTS OFF CACHE STRING "" FORCE)
set(BUILD_BENCHMARK OFF CACHE STRING "" FORCE)
FetchContent_Declare(inja URL ${INJA_URL} URL_HASH MD5=${INJA_URL_HASH})

FetchContent_MakeAvailable(json inja)

add_tablegen(oneflow_tblgen llvm tablegen.cpp op_schema_emitter.cpp)

if(LLVM_ENABLE_OBJLIB)
  set(OF_TBLGEN_TARGET obj.oneflow_tblgen)
else()
  set(OF_TBLGEN_TARGET oneflow_tblgen)
endif()

target_link_libraries(${OF_TBLGEN_TARGET} PRIVATE nlohmann_json::nlohmann_json pantor::inja)

install(TARGETS oneflow_tblgen LLVMTableGen LLVMDemangle LLVMSupport COMPONENT OneFlowTableGen
        LIBRARY DESTINATION lib)
add_custom_target(
  install-oneflow-tblgen DEPENDS oneflow_tblgen
  COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=OneFlowTableGen -P
          "${CMAKE_BINARY_DIR}/cmake_install.cmake")
