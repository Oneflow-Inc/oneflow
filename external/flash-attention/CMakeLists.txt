include(FetchContent)
FetchContent_Declare(
        cutlass
        URL ${CUTLASS_URL}
        URL_HASH MD5=${CUTLASS_MD5}
)
FetchContent_Populate(cutlass)

add_library(cutlass_headers INTERFACE)
set_target_properties(cutlass_headers PROPERTIES  INTERFACE_INCLUDE_DIRECTORIES ${cutlass_SOURCE_DIR}/include)

FetchContent_Declare(
        flash-attention
        URL ${FLASH_ATTENTION_URL}
        URL_HASH MD5=${FLASH_ATTENTION_MD5}
)
FetchContent_Populate(flash-attention)
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

add_library(of_flash_attention ${flash-attention_SOURCE_DIR}/csrc/flash_attn/src/fmha_fprop_fp16_kernel.sm80.cu ${flash-attention_SOURCE_DIR}/csrc/flash_attn/src/fmha_dgrad_fp16_kernel_loop.sm80.cu)
target_compile_options(of_flash_attention PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --expt-relaxed-constexpr --use_fast_math
                       >)
set_target_properties(of_flash_attention PROPERTIES CUDA_ARCHITECTURES "75;80;86")
target_include_directories(of_flash_attention PUBLIC $<BUILD_INTERFACE:${flash-attention_SOURCE_DIR}/csrc/flash_attn/src/>)
target_include_directories(of_flash_attention PRIVATE ${cutlass_SOURCE_DIR}/include)
