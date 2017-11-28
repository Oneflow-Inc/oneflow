#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/convolution_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {
template<typename T>
__global__ void Im2ColGpuKernel(const int n, const T* data_im, const int height,
                                const int width, const int kernel_h,
                                const int kernel_w, const int pad_h,
                                const int pad_w, const int stride_h,
                                const int stride_w, const int dilation_h,
                                const int dilation_w, const int height_col,
                                const int width_col, T* data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    T* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const T* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                ? data_im_ptr[i * dilation_h * width + j * dilation_w]
                : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template<typename T>
__global__ void Col2ImGpuKernel(const int n, const T* data_col,
                                const int height, const int width,
                                const int channels, const int kernel_h,
                                const int kernel_w, const int pad_h,
                                const int pad_w, const int stride_h,
                                const int stride_w, const int dilation_h,
                                const int dilation_w, const int height_col,
                                const int width_col, T* data_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    T val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index =
              (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col + h_col)
                  * width_col
              + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

}  // namespace

template<typename T>
class ConvolutionKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionKernelUtil);
  ConvolutionKernelUtil() = delete;
  static void Im2Col(const KernelCtx& ctx, const T* data_im, const int channels,
                     const int height, const int width, const int kernel_h,
                     const int kernel_w, const int pad_h, const int pad_w,
                     const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w, T* data_col) {
    int height_col =
        (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col =
        (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;
    Im2ColGpuKernel<T>
        <<<BlocksNum4ThreadsNum(num_kernels), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
            pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
            width_col, data_col);
  }

  static void Col2Im(const KernelCtx& ctx, const T* data_col,
                     const int channels, const int height, const int width,
                     const int kernel_h, const int kernel_w, const int pad_h,
                     const int pad_w, const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w, T* data_im) {
    int height_col =
        (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col =
        (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height * width;
    Col2ImGpuKernel<T>
        <<<BlocksNum4ThreadsNum(num_kernels), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            height_col, width_col, data_im);
  }
};

template<typename T>
CudnnConvolutionKernel<DeviceType::kGPU, T>::CudnnConvolutionKernel() {
  CudaCheck(cudnnCreateTensorDescriptor(&this->in_desc_));
  CudaCheck(cudnnCreateTensorDescriptor(&this->out_desc_));
  CudaCheck(cudnnCreateFilterDescriptor(&this->weight_desc_));
  CudaCheck(cudnnCreateConvolutionDescriptor(&this->conv_desc_));
}

template<typename T>
CudnnConvolutionKernel<DeviceType::kGPU, T>::~CudnnConvolutionKernel() {
  CudaCheck(cudnnDestroyTensorDescriptor(this->in_desc_));
  CudaCheck(cudnnDestroyTensorDescriptor(this->out_desc_));
  CudaCheck(cudnnDestroyFilterDescriptor(this->weight_desc_));
  CudaCheck(cudnnDestroyConvolutionDescriptor(this->conv_desc_));
  if (this->op_conf().convolution_conf().has_bias_term()) {
    CudaCheck(cudnnDestroyTensorDescriptor(this->bias_desc_));
  }
}

template<typename T>
void CudnnConvolutionKernel<DeviceType::kGPU, T>::VirtualKernelInit(
    bool is_forward, const ParallelContext* parallel_ctx) {
  auto conv_conf = this->op_conf().convolution_conf();
  CudaCheck(cudnnSetConvolution2dDescriptor(
      this->conv_desc_, conv_conf.pad_h(), conv_conf.pad_w(),
      conv_conf.stride_h(), conv_conf.stride_w(), 1, 1, CUDNN_CROSS_CORRELATION,
      CudnnDataType<T>::type));
  if (this->op_conf().convolution_conf().has_bias_term()) {
    CudaCheck(cudnnCreateTensorDescriptor(&this->bias_desc_));
  }
}

template<typename T>
void CudnnConvolutionKernel<DeviceType::kGPU, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto conv_conf = this->op_conf().convolution_conf();
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* fwd_workspace = BnInOp2Blob("fwd_workspace");

  CudaCheck(cudnnSetTensor4dDescriptor(
      this->in_desc_, CUDNN_TENSOR_NCHW, CudnnDataType<T>::type,
      in_blob->shape().At(0), in_blob->shape().At(1), in_blob->shape().At(2),
      in_blob->shape().At(3)));
  CudaCheck(cudnnSetTensor4dDescriptor(
      this->out_desc_, CUDNN_TENSOR_NCHW, CudnnDataType<T>::type,
      out_blob->shape().At(0), out_blob->shape().At(1), out_blob->shape().At(2),
      out_blob->shape().At(3)));
  CudaCheck(cudnnSetFilter4dDescriptor(
      this->weight_desc_, CudnnDataType<T>::type, CUDNN_TENSOR_NCHW,
      out_blob->shape().At(1), in_blob->shape().At(1), conv_conf.kernel_h(),
      conv_conf.kernel_w()));

  cudnnConvolutionFwdAlgo_t cudnn_fwd_algo =
      (cudnnConvolutionFwdAlgo_t)(conv_conf.cudnn_fwd_algo());
  CudaCheck(cudnnConvolutionForward(
      ctx.device_ctx->cudnn_handle(), CudnnDataType<T>::one, this->in_desc_,
      in_blob->dptr<T>(), this->weight_desc_, weight_blob->dptr<T>(),
      this->conv_desc_, cudnn_fwd_algo, fwd_workspace->mut_dptr<T>(),
      fwd_workspace->shape().At(0), CudnnDataType<T>::zero, this->out_desc_,
      out_blob->mut_dptr<T>()));

  if (this->op_conf().convolution_conf().has_bias_term()) {
    CudaCheck(cudnnSetTensor4dDescriptor(this->bias_desc_, CUDNN_TENSOR_NCHW,
                                         CudnnDataType<T>::type, 1,
                                         out_blob->shape().At(1), 1, 1));
    const Blob* bias_blob = BnInOp2Blob("bias");
    CudaCheck(cudnnAddTensor(ctx.device_ctx->cudnn_handle(),
                             CudnnDataType<T>::one, this->bias_desc_,
                             bias_blob->dptr<T>(), CudnnDataType<T>::one,
                             this->out_desc_, out_blob->mut_dptr<T>()));
  }
}

template<typename T>
void CudnnConvolutionKernel<DeviceType::kGPU, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto conv_conf = this->op_conf().convolution_conf();
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_blob = BnInOp2Blob("out");
  Blob* bwd_workspace = BnInOp2Blob("bwd_workspace");

  // compute bias diff
  if (this->op_conf().convolution_conf().has_bias_term()) {
    Blob* bias_diff_blob = BnInOp2Blob("bias_diff");
    Memset<DeviceType::kGPU>(ctx.device_ctx, bias_diff_blob->mut_dptr(), 0,
                             bias_diff_blob->ByteSizeOfDataContentField());

    CudaCheck(cudnnConvolutionBackwardBias(
        ctx.device_ctx->cudnn_handle(), CudnnDataType<T>::one, this->out_desc_,
        out_diff_blob->dptr<T>(), CudnnDataType<T>::one, this->bias_desc_,
        bias_diff_blob->mut_dptr<T>()));
  }

  // compute weight diff
  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");
  Memset<DeviceType::kGPU>(ctx.device_ctx, weight_diff_blob->mut_dptr(), 0,
                           weight_diff_blob->ByteSizeOfDataContentField());

  cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_weight_algo =
      (cudnnConvolutionBwdFilterAlgo_t)(conv_conf.cudnn_bwd_weight_algo());
  CudaCheck(cudnnConvolutionBackwardFilter(
      ctx.device_ctx->cudnn_handle(), CudnnDataType<T>::one, this->in_desc_,
      in_blob->dptr<T>(), this->out_desc_, out_diff_blob->dptr<T>(),
      this->conv_desc_, cudnn_bwd_weight_algo, bwd_workspace->mut_dptr<T>(),
      bwd_workspace->shape().At(0), CudnnDataType<T>::one, this->weight_desc_,
      weight_diff_blob->mut_dptr<T>()));

  // compute in diff
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<DeviceType::kGPU>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                           in_diff_blob->ByteSizeOfDataContentField());

  cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo =
      (cudnnConvolutionBwdDataAlgo_t)(conv_conf.cudnn_bwd_data_algo());
  CudaCheck(cudnnConvolutionBackwardData(
      ctx.device_ctx->cudnn_handle(), CudnnDataType<T>::one, this->weight_desc_,
      weight_blob->dptr<T>(), this->out_desc_, out_diff_blob->dptr<T>(),
      this->conv_desc_, cudnn_bwd_data_algo, bwd_workspace->mut_dptr<T>(),
      bwd_workspace->shape().At(0), CudnnDataType<T>::zero, this->in_desc_,
      in_diff_blob->mut_dptr<T>()));
}

template<typename T>
void CudnnConvolutionKernel<DeviceType::kGPU, T>::InitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<DeviceType::kGPU, T>::FillWithProperConf(
      ctx.device_ctx,
      OF_PB_POINTER_GET(this->op_conf().convolution_conf(), weight_fill),
      random_seed_gen(), BnInOp2Blob("weight"));

  if (this->op_conf().convolution_conf().has_bias_term()) {
    KernelUtil<DeviceType::kGPU, T>::FillWithProperConf(
        ctx.device_ctx,
        OF_PB_POINTER_GET(this->op_conf().convolution_conf(), bias_fill),
        random_seed_gen(), BnInOp2Blob("bias"));
  }
}

template<typename T>
void CudnnConvolutionKernel<DeviceType::kGPU, T>::InitModelBlobsWithDir(
    const KernelCtx& ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* weight_blob = BnInOp2Blob("weight");
  int32_t dim_num = this->op_conf().convolution_conf().out_num();
  KernelUtil<DeviceType::kGPU, T>::FillWithModelDir(
      ctx.device_ctx, part_id, part_num, model_load_dir, weight_blob, "weight",
      dim_num, weight_blob->shape().Count(1));
  if (this->op_conf().convolution_conf().has_bias_term()) {
    KernelUtil<DeviceType::kGPU, T>::FillWithModelDir(
        ctx.device_ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias"),
        "bias", dim_num, 1);
  }
}

template<typename T>
void CudnnConvolutionKernel<DeviceType::kGPU, T>::InitModelTmpBlobs(
    const KernelCtx& ctx, const ParallelContext* parallel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->op_conf().convolution_conf().has_bias_term()) {
    FillConf bias_multiplier_fill_conf;
    bias_multiplier_fill_conf.mutable_constant_conf()->set_value(1.0f);
    KernelUtil<DeviceType::kGPU, T>::Fill(ctx.device_ctx,
                                          bias_multiplier_fill_conf, 0,
                                          BnInOp2Blob("bias_multiplier"));
  }
}

#ifdef USE_CUDNN
#define INSTANTIATE_CONVOLUTION_KERNEL(type_cpp, type_proto) \
  template class CudnnConvolutionKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTAITE_CONVOLUTION_KERNEL, FLOATING_DATA_TYPE_SEQ)
#endif  // USE_CUDNN

#define INSTANTIATE_CONVOLUTION_KERNEL_UTIL(type_cpp, type_proto) \
  template class ConvolutionKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONVOLUTION_KERNEL_UTIL,
                     FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
