#include "oneflow/user/kernels/fft_kernel_util.h"
#include "oneflow/core/common/shape.h"
#include "pocketfftplan.h"

namespace oneflow{

template<typename IN, typename OUT, typename fct_type>
struct FftC2CKernelUtil<DeviceType::kCPU, IN, OUT, fct_type>{
    static void FftC2CForward(ep::Stream* stream, IN* data_in, OUT* data_out, const Shape& input_shape, 
                              const Shape& output_shape, bool forward, const std::vector<int64_t>& dims, fft_norm_mode normalization){
        
    PocketFFtParams<IN, OUT, fct_type> params(input_shape, output_shape, dims, forward,
                                    compute_fct<fct_type>(input_shape, dims, normalization) /*1.f*/,
                                    FFT_EXCUTETYPE::C2C);
    PocketFFtConfig<IN, OUT, fct_type> config(params);
    config.excute(data_in, data_out);
    }
};

// OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_FFTC2C_KERNEL_UTIL, (DeviceType::kCPU),
//                                  COMPLEX_DATA_TYPE_SEQ, COMPLEX_DATA_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ);
INSTANTIATE_FFTC2C_KERNEL_UTIL((DeviceType::kCPU), std::complex<float>, std::complex<float>, float);
INSTANTIATE_FFTC2C_KERNEL_UTIL((DeviceType::kCPU), std::complex<double>, std::complex<double>, double);
}   // namespace oneflow