// TODO: move the content of .cpp to .h

#if defined(WITH_CUDA)
#include <cuda_fp16.h>

namespace oneflow {

template<typename T>
struct IsFloat16;

template<>
struct IsFloat16<half> : std::integral_constant<bool, true> {};

}

#endif
