#include "oneflow/user/kernels/fft_kernel_util.h"


namespace oneflow{

template<typename T, typename IDX>
OF_DEVICE_FUNC void FftForwardCompute(IDX elem_num, const T* x_ptr, const T* other_ptr, T* y_ptr) {
    FOR_RANGE(IDX, i, 0, elem_num){
        y_ptr[i] = x_ptr[i] * other_ptr[i];
    }
}

template<typename T, typename IDX>
struct FftKernelUtil<DeviceType::kCPU, T, IDX>{
    static void FftForward(ep::Stream* stream, const IDX elem_num, 
                           const T* x, const T* other, T* y){
        FftForwardCompute<T, IDX>(elem_num, x, other, y);
    }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_FFT_KERNEL_UTIL, (DeviceType::kCPU),
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_FFT_KERNEL_UTIL, (DeviceType::kCPU),
                                 SIGNED_INT_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_FFT_KERNEL_UTIL, (DeviceType::kCPU),
                                 UNSIGNED_INT_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);

}   // namespace oneflow