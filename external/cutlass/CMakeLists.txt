include(FetchContent)
FetchContent_Declare(
        cutlass
        URL ${CUTLASS_URL}
        URL_HASH MD5=${CUTLASS_MD5}
)
FetchContent_Populate(cutlass)

add_library(cutlass_headers INTERFACE)
set_property(TARGET cutlass_headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  $<BUILD_INTERFACE:${cutlass_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
add_library(cutlass_fmha_headers INTERFACE)
set_property(TARGET cutlass_fmha_headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  $<BUILD_INTERFACE:${cutlass_SOURCE_DIR}/examples/42_fused_multi_head_attention>
  $<INSTALL_INTERFACE:include>
)
set_property(GLOBAL PROPERTY CUTLASS_FMHA_HEADERS_DIR ${cutlass_SOURCE_DIR}/examples/42_fused_multi_head_attention)
install(TARGETS cutlass_headers;cutlass_fmha_headers EXPORT oneflow DESTINATION include)
