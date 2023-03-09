#include <cstdint>
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/kernels/fft_kernel_util.h"

namespace oneflow{

namespace {

constexpr int kBlockSize = cuda::elementwise::kBlockSize;

int GetMinThreadNum(const int64_t elem_num) { return std::min<int64_t>(elem_num, kBlockSize); }

int GetNumBlocks(int32_t elem_cnt) {
  int num_blocks = 0;
  OF_CUDA_CHECK(cuda::elementwise::GetNumBlocks(elem_cnt, &num_blocks));
  return num_blocks;
}

}  // namespace

template<typename T, typename IDX>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAFftForward(const IDX elem_num, const T* x, const T* other, T* y)
    {
        CUDA_1D_KERNEL_LOOP(i, elem_num){
            y[i] = x[i] * other[i];
        }
    }


template<typename T, typename IDX>
struct FftKernelUtil<DeviceType::kCUDA, T, IDX> final {
    static void FftForward(ep::Stream* stream, const IDX elem_num, 
                           const T* x, const T* other, T* y){
        DoCUDAFftForward<T, IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
                                elem_num, x, other, y);
    }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_FFT_KERNEL_UTIL, (DeviceType::kCUDA),
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_FFT_KERNEL_UTIL, (DeviceType::kCUDA),
                                 SIGNED_INT_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_FFT_KERNEL_UTIL, (DeviceType::kCUDA),
                                 UNSIGNED_INT_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
}   //  namespace oneflow