#include "oneflow/user/kernels/fft_kernel_util.h"
#include "oneflow/core/common/shape.h"
#include "pocketfftplan.h"

namespace oneflow{

template<typename IN, typename OUT, typename dtype>
struct FftC2CKernelUtil<DeviceType::kCPU, IN, OUT, dtype>{
    static void FftC2CForward(ep::Stream* stream, IN* data_in, OUT* data_out, 
                              const Shape& input_shape, const Shape& output_shape, 
                              const Stride& input_stride, const Stride& output_stride,
                              bool forward, const std::vector<int64_t>& dims, fft_norm_mode normalization){
        
    PocketFFtParams<dtype> params(input_shape, output_shape, input_stride, output_stride, dims, forward,
                                    compute_fct<dtype>(input_shape, dims, normalization) /*1.f*/,
                                    FFT_EXCUTETYPE::C2C);
    PocketFFtConfig<dtype> config(params);
    config.excute(data_in, data_out);
    }
};


template<typename IN, typename OUT, typename dtype>
struct FftR2CKernelUtil<DeviceType::kCPU, IN, OUT, dtype>{
    static void FftR2CForward(ep::Stream* stream, IN* data_in, OUT* data_out, 
                              const Shape& input_shape, const Shape& output_shape, 
                              const Stride& input_stride, const Stride& output_stride,
                              bool forward, const std::vector<int64_t>& dims,
                              fft_norm_mode normalization){
    // get temp buffer ? or use out, must be sure `out` is contiguos?

    // get last dim half size

    // do r2c, get half size fft out
    PocketFFtParams<dtype> params(input_shape, output_shape, input_stride, output_stride, dims, forward,
                                    compute_fct<dtype>(input_shape, dims, normalization) /*1.f*/,
                                    FFT_EXCUTETYPE::R2C);
    PocketFFtConfig<dtype> config(params);
    config.excute(data_in, data_out);

    // convert_to_doublesized
    }
};

// OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_FFTC2C_KERNEL_UTIL, (DeviceType::kCPU),
//                                  COMPLEX_DATA_TYPE_SEQ, COMPLEX_DATA_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ);
INSTANTIATE_FFTC2C_KERNEL_UTIL((DeviceType::kCPU), std::complex<float>, std::complex<float>, float);
INSTANTIATE_FFTC2C_KERNEL_UTIL((DeviceType::kCPU), std::complex<double>, std::complex<double>, double);
}   // namespace oneflow