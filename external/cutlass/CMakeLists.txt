include(FetchContent)
FetchContent_Declare(
        cutlass
        URL ${CUTLASS_URL}
        URL_HASH MD5=${CUTLASS_MD5}
)
FetchContent_Populate(cutlass)

add_library(cutlass_headers INTERFACE)
set_target_properties(cutlass_headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${cutlass_SOURCE_DIR}/include)
add_library(cutlass_fmha_headers INTERFACE)
set_target_properties(cutlass_fmha_headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
	${cutlass_SOURCE_DIR}/examples/42_fused_multi_head_attention)
