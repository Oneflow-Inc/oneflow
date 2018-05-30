#include "oneflow/core/kernel/conv_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  if (this->UseCudnnOnGpu()) {
    KernelInitWithCudnn(parallel_ctx);
  } else {
    KernelInitWithoutCudnn(parallel_ctx);
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::DoForwardDataContent(
    DeviceCtx* device_ctx, const Blob* in_blob, const Blob* weight_blob, Blob* out_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->UseCudnnOnGpu()) {
    DoForwardDataContentWithCudnn(device_ctx, in_blob, weight_blob, out_blob, BnInOp2Blob);
  } else {
    DoForwardDataContentWithoutCudnn(device_ctx, in_blob, weight_blob, out_blob, BnInOp2Blob);
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::WeightBackward(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, const Blob* in_blob, Blob* weight_diff_blob,
    Blob* in_diff_blob, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->UseCudnnOnGpu()) {
    WeightBackwardWithCudnn(device_ctx, out_diff_blob, in_blob, weight_diff_blob, in_diff_blob,
                            BnInOp2Blob);
  } else {
    WeightBackwardWithoutCudnn(device_ctx, out_diff_blob, in_blob, weight_diff_blob, in_diff_blob,
                               BnInOp2Blob);
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::BiasBackward(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, Blob* bias_diff_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->UseCudnnOnGpu()) {
    BiasBackwardWithCudnn(device_ctx, out_diff_blob, bias_diff_blob, BnInOp2Blob);
  } else {
    BiasBackwardWithoutCudnn(device_ctx, out_diff_blob, bias_diff_blob, BnInOp2Blob);
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::KernelInitWithCudnn(const ParallelContext* parallel_ctx) {
  Shape in_shape(this->GetConvKernelConf().in());
  Shape out_shape(this->GetConvKernelConf().out());
  Shape weight_shape(this->GetConvKernelConf().weight());

  const std::string& data_format =
      this->template GetValFromCustomizedOpConf<std::string>("data_format");
  this->in_desc_.reset(new CudnnTensorDesc(GetDataType<T>::value, in_shape, data_format));
  this->out_desc_.reset(new CudnnTensorDesc(GetDataType<T>::value, out_shape, data_format));
  this->filter_desc_.reset(new CudnnFilterDesc(GetDataType<T>::value, weight_shape, data_format));
  this->conv_desc_.reset(
      new CudnnConvDesc(GetDataType<T>::value, in_shape, this->GetCustomizedOpConf()));

  if (this->template GetValFromCustomizedOpConf<bool>("use_bias")) {
    int32_t filters = this->template GetValFromCustomizedOpConf<int32_t>("filters");
    if ((this->OpKernelDim() == 1) || (this->OpKernelDim() == 2)) {
      if (data_format == "channels_first") {
        this->bias_desc_.reset(
            new CudnnTensorDesc(CUDNN_TENSOR_NCHW, GetDataType<T>::value, 1, filters, 1, 1));
      } else if (data_format == "channels_last") {
        if (GetDataType<T>::value == DataType::kDouble) {
          LOG(FATAL) << "CUDNN 1d & 2d support channels last only if data type "
                        "is float";
        }
        this->bias_desc_.reset(
            new CudnnTensorDesc(CUDNN_TENSOR_NHWC, GetDataType<T>::value, 1, filters, 1, 1));
      } else {
        UNIMPLEMENTED();
      }
    } else {
      if (data_format == "channels_last") {
        LOG(FATAL) << "CUDNN Nd API only support channels first";
      }
      std::vector<int32_t> bias_dim(this->OpKernelDim() + 2, 1);
      std::vector<int32_t> stride_of_bias_tensor(this->OpKernelDim() + 2, 1);
      bias_dim[1] = filters;
      stride_of_bias_tensor[0] = filters;
      this->bias_desc_.reset(new CudnnTensorDesc(GetDataType<T>::value, this->OpKernelDim() + 2,
                                                 bias_dim.data(), stride_of_bias_tensor.data()));
    }
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::DoForwardDataContentWithCudnn(
    DeviceCtx* device_ctx, const Blob* in_blob, const Blob* weight_blob, Blob* out_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CudaCheck(cudnnConvolutionForward(
      device_ctx->cudnn_handle(), OnePtr<T>::value, this->in_desc_->Get(), in_blob->dptr<T>(),
      this->filter_desc_->Get(), weight_blob->dptr<T>(), this->conv_desc_->Get(),
      static_cast<cudnnConvolutionFwdAlgo_t>(this->GetConvKernelConf().cudnn_fwd_algo()),
      device_ctx->buf_ptr(), device_ctx->buf_size(), ZeroPtr<T>::value, this->out_desc_->Get(),
      out_blob->mut_dptr<T>()));

  if (this->template GetValFromCustomizedOpConf<bool>("use_bias")) {
    const Blob* bias = BnInOp2Blob("bias");
    CudaCheck(cudnnAddTensor(device_ctx->cudnn_handle(), OnePtr<T>::value, this->bias_desc_->Get(),
                             bias->dptr<T>(), OnePtr<T>::value, this->out_desc_->Get(),
                             out_blob->mut_dptr<T>()));
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::WeightBackwardWithCudnn(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, const Blob* in_blob, Blob* weight_diff_blob,
    Blob* in_diff_blob, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* weight_blob = BnInOp2Blob("weight");
  CudaCheck(cudnnConvolutionBackwardFilter(
      device_ctx->cudnn_handle(), OnePtr<T>::value, this->in_desc_->Get(), in_blob->dptr<T>(),
      this->out_desc_->Get(), out_diff_blob->dptr<T>(), this->conv_desc_->Get(),
      static_cast<cudnnConvolutionBwdFilterAlgo_t>(
          this->GetConvKernelConf().cudnn_bwd_filter_algo()),
      device_ctx->buf_ptr(), device_ctx->buf_size(), ZeroPtr<T>::value, this->filter_desc_->Get(),
      weight_diff_blob->mut_dptr<T>()));

  if (in_diff_blob != nullptr) {
    CudaCheck(cudnnConvolutionBackwardData(
        device_ctx->cudnn_handle(), OnePtr<T>::value, this->filter_desc_->Get(),
        weight_blob->dptr<T>(), this->out_desc_->Get(), out_diff_blob->dptr<T>(),
        this->conv_desc_->Get(),
        static_cast<cudnnConvolutionBwdDataAlgo_t>(this->GetConvKernelConf().cudnn_bwd_data_algo()),
        device_ctx->buf_ptr(), device_ctx->buf_size(), ZeroPtr<T>::value, this->in_desc_->Get(),
        in_diff_blob->mut_dptr<T>()));
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::BiasBackwardWithCudnn(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, Blob* bias_diff_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CudaCheck(cudnnConvolutionBackwardBias(device_ctx->cudnn_handle(), OnePtr<T>::value,
                                         this->out_desc_->Get(), out_diff_blob->dptr<T>(),
                                         ZeroPtr<T>::value, this->bias_desc_->Get(),
                                         bias_diff_blob->mut_dptr<T>()));
}

__device__ void InitSharedArrays(const int im_d, const int im_h, const int im_w, const int kernel_d,
                                 const int kernel_h, const int kernel_w, const int out_d,
                                 const int out_h, const int out_w, const int stride_d,
                                 const int stride_h, const int stride_w, const int dilation_d,
                                 const int dilation_h, const int dilation_w, const int pad_d,
                                 const int pad_h, const int pad_w, int* shared_im,
                                 int* shared_kernel, int* shared_out, int* shared_stride,
                                 int* shared_dilation, int* shared_pad) {
  if (threadIdx.x == 0) {
    shared_im[0] = im_d;
    shared_im[1] = im_h;
    shared_im[2] = im_w;
    shared_kernel[0] = kernel_d;
    shared_kernel[1] = kernel_h;
    shared_kernel[2] = kernel_w;
    shared_out[0] = out_d;
    shared_out[1] = out_h;
    shared_out[2] = out_w;
    shared_stride[0] = stride_d;
    shared_stride[1] = stride_h;
    shared_stride[2] = stride_w;
    shared_dilation[0] = dilation_d;
    shared_dilation[1] = dilation_h;
    shared_dilation[2] = dilation_w;
    shared_pad[0] = pad_d;
    shared_pad[1] = pad_h;
    shared_pad[2] = pad_w;
  }
  __syncthreads();
}

template<typename T>
__global__ void NCDHWIm2ColGpu(const int n, const T* im_dptr, const int channel, const int im_d,
                               const int im_h, const int im_w, const int kernel_d,
                               const int kernel_h, const int kernel_w, const int out_d,
                               const int out_h, const int out_w, const int stride_d,
                               const int stride_h, const int stride_w, const int dilation_rate_d,
                               const int dilation_rate_h, const int dilation_rate_w,
                               const int padding_before_d, const int padding_before_h,
                               const int padding_before_w, T* col_buf_dptr) {
  const int dim_num = 3;
  __shared__ int shared_im[dim_num];
  __shared__ int shared_kernel[dim_num];
  __shared__ int shared_out[dim_num];
  __shared__ int shared_stride[dim_num];
  __shared__ int shared_dilation[dim_num];
  __shared__ int shared_pad[dim_num];
  InitSharedArrays(im_d, im_h, im_w, kernel_d, kernel_h, kernel_w, out_d, out_h, out_w, stride_d,
                   stride_h, stride_w, dilation_rate_d, dilation_rate_h, dilation_rate_w,
                   padding_before_d, padding_before_h, padding_before_w, shared_im, shared_kernel,
                   shared_out, shared_stride, shared_dilation, shared_pad);

  int out_size = 1;
  for (int i = 0; i < dim_num; ++i) { out_size *= shared_out[i]; }
  int kernel_index[dim_num];
  int out_index[dim_num];
  int channel_index;
  int im_index[dim_num];
  CUDA_1D_KERNEL_LOOP(index, n) {
    // calc kernel_/out_/channel_index
    int row_offset = index / out_size;  // row_dim of col_buf: channel*kd*kh*kw
    int col_offset = index % out_size;  // col_dim of col_buf: od*oh*ow
    for (int i = dim_num - 1; i >= 0; --i) {
      out_index[i] = col_offset % shared_out[i];
      col_offset /= shared_out[i];
      kernel_index[i] = row_offset % shared_kernel[i];
      row_offset /= shared_kernel[i];
    }
    channel_index = row_offset;

    // calc im_index
    bool is_im_index_valid = true;
    for (int i = 0; i < dim_num; ++i) {
      im_index[i] =
          kernel_index[i] * shared_dilation[i] - shared_pad[i] + out_index[i] * shared_stride[i];
      if (im_index[i] < 0 || im_index[i] >= shared_im[i]) {
        is_im_index_valid = false;
        break;
      }
    }

    // write into col_buf
    if (is_im_index_valid) {
      // calc im_offset
      int im_offset = channel_index;
      for (int i = 0; i < dim_num; ++i) {
        im_offset *= shared_im[i];
        im_offset += im_index[i];
      }
      col_buf_dptr[index] = im_dptr[im_offset];
    } else {
      col_buf_dptr[index] = 0;
    }
  }
}

template<typename T>
__global__ void NCDHWCol2ImGpu(const int n, const T* col_buf_dptr, const int channel,
                               const int im_d, const int im_h, const int im_w, const int kernel_d,
                               const int kernel_h, const int kernel_w, const int out_d,
                               const int out_h, const int out_w, const int stride_d,
                               const int stride_h, const int stride_w, const int dilation_rate_d,
                               const int dilation_rate_h, const int dilation_rate_w,
                               const int padding_before_d, const int padding_before_h,
                               const int padding_before_w, T* im_diff_dptr) {
  const int dim_num = 3;
  __shared__ int shared_im[dim_num];
  __shared__ int shared_kernel[dim_num];
  __shared__ int shared_out[dim_num];
  __shared__ int shared_stride[dim_num];
  __shared__ int shared_dilation[dim_num];
  __shared__ int shared_pad[dim_num];
  InitSharedArrays(im_d, im_h, im_w, kernel_d, kernel_h, kernel_w, out_d, out_h, out_w, stride_d,
                   stride_h, stride_w, dilation_rate_d, dilation_rate_h, dilation_rate_w,
                   padding_before_d, padding_before_h, padding_before_w, shared_im, shared_kernel,
                   shared_out, shared_stride, shared_dilation, shared_pad);

  int kernel_index[dim_num];
  int channel_index;
  int im_index[dim_num];
  int out_begin[dim_num];
  int out_end[dim_num];
  int out_index[dim_num];
  CUDA_1D_KERNEL_LOOP(index, n) {
    // calc im_/channel_index
    int im_offset = index;
    for (int i = dim_num - 1; i >= 0; --i) {
      im_index[i] = im_offset % shared_im[i] + shared_pad[i];
      im_offset /= shared_im[i];
    }
    channel_index = im_offset;

    // calc the out_range of this im element
    bool is_in_dim_wrong = false;
    for (int i = 0; i < dim_num; ++i) {
      const int kernel_extent = shared_dilation[i] * (shared_kernel[i] - 1) + 1;
      if (im_index[i] < kernel_extent) {
        out_begin[i] = 0;
      } else {
        // original equation: ((im_index[i]-kernel_extent+1)+(stride[i]-1))/stride[i]
        out_begin[i] = (im_index[i] - kernel_extent) / shared_stride[i] + 1;
      }
      out_end[i] = min(im_index[i] / shared_stride[i] + 1, shared_out[i]);
      out_index[i] = out_begin[i];

      if (out_begin[i] >= out_end[i]) {  // for those im elements not chosen by kernel
        is_in_dim_wrong = true;
        break;
      }
    }
    if (is_in_dim_wrong) {
      im_diff_dptr[index] = 0;
      continue;
    }

    T val = 0;
    while (true) {
      bool is_skip = false;
      // calc kernel_index
      for (int i = 0; i < dim_num; ++i) {
        kernel_index[i] = im_index[i] - out_index[i] * shared_stride[i];
        if (kernel_index[i] % shared_dilation[i] == 0) {
          kernel_index[i] /= shared_dilation[i];
        } else {
          is_skip = true;
          break;
        }
      }

      // cal col_buf_offset
      if (is_skip == false) {
        int col_buf_offset = channel_index;
        for (int i = 0; i < dim_num; ++i) {
          col_buf_offset *= shared_kernel[i];
          col_buf_offset += kernel_index[i];
        }
        for (int i = 0; i < dim_num; ++i) {
          col_buf_offset *= shared_out[i];
          col_buf_offset += out_index[i];
        }
        val += col_buf_dptr[col_buf_offset];
      }

      // iter next out_index[]
      bool is_iter_completed = true;
      for (int i = dim_num - 1; i >= 0; --i) {
        if (out_index[i] == out_end[i] - 1) {
          out_index[i] = out_begin[i];
        } else {
          out_index[i] += 1;
          is_iter_completed = false;
          break;
        }
      }
      if (is_iter_completed) { break; }
    }
    im_diff_dptr[index] = val;
  }
}

template<typename T>
__global__ void NDHWCIm2ColGpu(const int n, const T* im_dptr, const int channel, const int im_d,
                               const int im_h, const int im_w, const int kernel_d,
                               const int kernel_h, const int kernel_w, const int out_d,
                               const int out_h, const int out_w, const int stride_d,
                               const int stride_h, const int stride_w, const int dilation_rate_d,
                               const int dilation_rate_h, const int dilation_rate_w,
                               const int padding_before_d, const int padding_before_h,
                               const int padding_before_w, T* col_buf_dptr) {
  const int dim_num = 3;
  __shared__ int shared_im[dim_num];
  __shared__ int shared_kernel[dim_num];
  __shared__ int shared_out[dim_num];
  __shared__ int shared_stride[dim_num];
  __shared__ int shared_dilation[dim_num];
  __shared__ int shared_pad[dim_num];
  InitSharedArrays(im_d, im_h, im_w, kernel_d, kernel_h, kernel_w, out_d, out_h, out_w, stride_d,
                   stride_h, stride_w, dilation_rate_d, dilation_rate_h, dilation_rate_w,
                   padding_before_d, padding_before_h, padding_before_w, shared_im, shared_kernel,
                   shared_out, shared_stride, shared_dilation, shared_pad);

  int out_size = 1;
  for (int i = 0; i < dim_num; ++i) { out_size *= shared_out[i]; }
  int kernel_index[dim_num];
  int out_index[dim_num];
  int channel_index;
  int im_index[dim_num];
  CUDA_1D_KERNEL_LOOP(index, n) {
    // calc kernel_/out_/channel_index
    int row_offset = index / out_size;  // row_dim of col_buf: kd*kh*kw*channel
    int col_offset = index % out_size;  // col_dim of col_buf: od*oh*ow
    channel_index = row_offset % channel;
    row_offset /= channel;
    for (int i = dim_num - 1; i >= 0; --i) {
      out_index[i] = col_offset % shared_out[i];
      col_offset /= shared_out[i];
      kernel_index[i] = row_offset % shared_kernel[i];
      row_offset /= shared_kernel[i];
    }
    assert(row_offset == 0);

    // calc im_index
    bool is_im_index_valid = true;
    for (int i = 0; i < dim_num; ++i) {
      im_index[i] =
          kernel_index[i] * shared_dilation[i] - shared_pad[i] + out_index[i] * shared_stride[i];
      if (im_index[i] < 0 || im_index[i] >= shared_im[i]) {
        is_im_index_valid = false;
        break;
      }
    }

    // write into col_buf
    if (is_im_index_valid) {
      // calc im_offset
      int im_offset = channel_index;
      for (int i = 0; i < dim_num; ++i) {
        im_offset *= shared_im[i];
        im_offset += im_index[i];
      }
      col_buf_dptr[index] = im_dptr[im_offset];
    } else {
      col_buf_dptr[index] = 0;
    }
  }
}

template<typename T>
__global__ void NDHWCCol2ImGpu(const int n, const T* col_buf_dptr, const int channel,
                               const int im_d, const int im_h, const int im_w, const int kernel_d,
                               const int kernel_h, const int kernel_w, const int out_d,
                               const int out_h, const int out_w, const int stride_d,
                               const int stride_h, const int stride_w, const int dilation_rate_d,
                               const int dilation_rate_h, const int dilation_rate_w,
                               const int padding_before_d, const int padding_before_h,
                               const int padding_before_w, T* im_diff_dptr) {
  const int dim_num = 3;
  __shared__ int shared_im[dim_num];
  __shared__ int shared_kernel[dim_num];
  __shared__ int shared_out[dim_num];
  __shared__ int shared_stride[dim_num];
  __shared__ int shared_dilation[dim_num];
  __shared__ int shared_pad[dim_num];
  InitSharedArrays(im_d, im_h, im_w, kernel_d, kernel_h, kernel_w, out_d, out_h, out_w, stride_d,
                   stride_h, stride_w, dilation_rate_d, dilation_rate_h, dilation_rate_w,
                   padding_before_d, padding_before_h, padding_before_w, shared_im, shared_kernel,
                   shared_out, shared_stride, shared_dilation, shared_pad);

  int kernel_index[dim_num];
  int channel_index;
  int im_index[dim_num];
  int out_begin[dim_num];
  int out_end[dim_num];
  int out_index[dim_num];
  CUDA_1D_KERNEL_LOOP(index, n) {
    // calc im_/channel_index
    int im_offset = index;
    channel_index = im_offset % channel;
    im_offset /= channel;
    for (int i = dim_num - 1; i >= 0; --i) {
      im_index[i] = im_offset % shared_im[i] + shared_pad[i];
      im_offset /= shared_im[i];
    }
    assert(im_offset == 0);

    // calc the out_range of this im element
    bool is_in_dim_wrong = false;
    for (int i = 0; i < dim_num; ++i) {
      const int kernel_extent = shared_dilation[i] * (shared_kernel[i] - 1) + 1;
      if (im_index[i] < kernel_extent) {
        out_begin[i] = 0;
      } else {
        // original equation: ((im_index[i]-kernel_extent+1)+(stride[i]-1))/stride[i]
        out_begin[i] = (im_index[i] - kernel_extent) / shared_stride[i] + 1;
      }
      out_end[i] = min(im_index[i] / shared_stride[i] + 1, shared_out[i]);
      out_index[i] = out_begin[i];

      if (out_begin[i] >= out_end[i]) {  // for those im elements not chosen by kernel
        is_in_dim_wrong = true;
        break;
      }
    }
    if (is_in_dim_wrong) {
      im_diff_dptr[index] = 0;
      continue;
    }

    T val = 0;
    while (true) {
      bool is_skip = false;
      // calc kernel_index
      for (int i = 0; i < dim_num; ++i) {
        kernel_index[i] = im_index[i] - out_index[i] * shared_stride[i];
        if (kernel_index[i] % shared_dilation[i] == 0) {
          kernel_index[i] /= shared_dilation[i];
        } else {
          is_skip = true;
          break;
        }
      }

      // cal col_buf_offset
      if (is_skip == false) {
        int col_buf_offset = 0;
        for (int i = 0; i < dim_num; ++i) {
          col_buf_offset *= shared_kernel[i];
          col_buf_offset += kernel_index[i];
        }
        col_buf_offset *= channel;
        col_buf_offset += channel_index;
        for (int i = 0; i < dim_num; ++i) {
          col_buf_offset *= shared_out[i];
          col_buf_offset += out_index[i];
        }
        val += col_buf_dptr[col_buf_offset];
      }

      // iter next out_index[]
      bool is_iter_completed = true;
      for (int i = dim_num - 1; i >= 0; --i) {
        if (out_index[i] == out_end[i] - 1) {
          out_index[i] = out_begin[i];
        } else {
          out_index[i] += 1;
          is_iter_completed = false;
          break;
        }
      }
      if (is_iter_completed) { break; }
    }
    im_diff_dptr[index] = val;
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::KernelInitWithoutCudnn(const ParallelContext* parallel_ctx) {
  const std::string& data_format =
      this->template GetValFromCustomizedOpConf<std::string>("data_format");
  if (data_format == "channels_first") {
    im2col_func_ = ConvKernelGpuUtil<T>::NCDHWIm2Col;
    col2im_func_ = ConvKernelGpuUtil<T>::NCDHWCol2Im;
    forward_func_ = KernelUtil<DeviceType::kGPU, T>::OFGemm;
    dhw_offset_ = 2;
    is_out_diff_need_trans_ = CblasNoTrans;
  } else {
    im2col_func_ = ConvKernelGpuUtil<T>::NDHWCIm2Col;
    col2im_func_ = ConvKernelGpuUtil<T>::NDHWCCol2Im;
    forward_func_ = KernelUtil<DeviceType::kGPU, T>::OFGemmTrans;
    dhw_offset_ = 1;
    is_out_diff_need_trans_ = CblasTrans;
  }
  in_shape_ = Shape(this->GetConvKernelConf().in());
  out_shape_ = Shape(this->GetConvKernelConf().out());
  weight_shape_ = Shape(this->GetConvKernelConf().weight());
  strides_ = this->GetConvKernelConf().strides().data();
  dilation_rate_ = this->GetConvKernelConf().dilation_rate().data();
  padding_before_ = this->GetConvKernelConf().pad_small_side().data();
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::DoForwardDataContentWithoutCudnn(
    DeviceCtx* device_ctx, const Blob* in_blob, const Blob* weight_blob, Blob* out_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FOR_RANGE(int64_t, i, 0, in_shape_.At(0)) {
    im2col_func_(device_ctx, GetImgDptr<T>(in_blob, i), in_shape_, weight_shape_, out_shape_,
                 strides_, dilation_rate_, padding_before_, static_cast<T*>(device_ctx->buf_ptr()));

    // col_buf is device_ctx->buf_ptr()
    // channels first: out = weight * col_buf
    // channels last:  out = (weight * col_buf)(T)
    forward_func_(device_ctx, CblasNoTrans, CblasNoTrans,
                  weight_shape_.At(0),                             // filter
                  out_shape_.Count(dhw_offset_, dhw_offset_ + 3),  // od * oh * ow
                  weight_shape_.Count(1),                          // ci * kd * kh * kw
                  static_cast<T>(1), weight_blob->dptr<T>(),
                  static_cast<const T*>(device_ctx->buf_ptr()), static_cast<T>(0),
                  GetImgMutDptr<T>(out_blob, i));
    if (this->template GetValFromCustomizedOpConf<bool>("use_bias")) {
      const Blob* bias_blob = BnInOp2Blob("bias");
      const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");
      // channels first:  out += bias * bias_mul
      // channels last:   out += (bias * bias_mul)(T)
      forward_func_(device_ctx, CblasNoTrans, CblasNoTrans,
                    weight_shape_.At(0),                             // filter
                    out_shape_.Count(dhw_offset_, dhw_offset_ + 3),  // od * oh * ow
                    1,                                               // 1
                    static_cast<T>(1), bias_blob->dptr<T>(), bias_mul_blob->dptr<T>(),
                    static_cast<T>(1), GetImgMutDptr<T>(out_blob, i));
    }
  }
}

template<typename T>
void ConvKernelGpuUtil<T>::NCDHWIm2Col(DeviceCtx* device_ctx, const T* in_dptr,
                                       const Shape& in_shape, const Shape& weight_shape,
                                       const Shape& out_shape, const int32_t* strides,
                                       const int32_t* dilation_rate, const int32_t* padding_before,
                                       T* col_buf_ptr) {
  int32_t col_buf_size = weight_shape.Count(1) * out_shape.Count(2);
  NCDHWIm2ColGpu<T><<<BlocksNum4ThreadsNum(col_buf_size), kCudaThreadsNumPerBlock, 0,
                      device_ctx->cuda_stream()>>>(
      col_buf_size, in_dptr, in_shape.At(1), in_shape.At(2), in_shape.At(3), in_shape.At(4),
      weight_shape.At(2), weight_shape.At(3), weight_shape.At(4), out_shape.At(2), out_shape.At(3),
      out_shape.At(4), strides[0], strides[1], strides[2], dilation_rate[0], dilation_rate[1],
      dilation_rate[2], padding_before[0], padding_before[1], padding_before[2], col_buf_ptr);
}

template<typename T>
void ConvKernelGpuUtil<T>::NCDHWCol2Im(DeviceCtx* device_ctx, const T* col_buf_dptr,
                                       const Shape& in_shape, const Shape& weight_shape,
                                       const Shape& out_shape, const int32_t* strides,
                                       const int32_t* dilation_rate, const int32_t* padding_before,
                                       T* in_diff_ptr) {
  int32_t im_size = in_shape.Count(1);
  NCDHWCol2ImGpu<T>
      <<<BlocksNum4ThreadsNum(im_size), kCudaThreadsNumPerBlock, 0, device_ctx->cuda_stream()>>>(
          im_size, col_buf_dptr, in_shape.At(1), in_shape.At(2), in_shape.At(3), in_shape.At(4),
          weight_shape.At(2), weight_shape.At(3), weight_shape.At(4), out_shape.At(2),
          out_shape.At(3), out_shape.At(4), strides[0], strides[1], strides[2], dilation_rate[0],
          dilation_rate[1], dilation_rate[2], padding_before[0], padding_before[1],
          padding_before[2], in_diff_ptr);
}

template<typename T>
void ConvKernelGpuUtil<T>::NDHWCIm2Col(DeviceCtx* device_ctx, const T* in_dptr,
                                       const Shape& in_shape, const Shape& weight_shape,
                                       const Shape& out_shape, const int32_t* strides,
                                       const int32_t* dilation_rate, const int32_t* padding_before,
                                       T* col_buf_ptr) {
  int32_t col_buf_size = weight_shape.Count(1) * out_shape.Count(2);
  NDHWCIm2ColGpu<T><<<BlocksNum4ThreadsNum(col_buf_size), kCudaThreadsNumPerBlock, 0,
                      device_ctx->cuda_stream()>>>(
      col_buf_size, in_dptr, in_shape.At(1), in_shape.At(2), in_shape.At(3), in_shape.At(4),
      weight_shape.At(2), weight_shape.At(3), weight_shape.At(4), out_shape.At(2), out_shape.At(3),
      out_shape.At(4), strides[0], strides[1], strides[2], dilation_rate[0], dilation_rate[1],
      dilation_rate[2], padding_before[0], padding_before[1], padding_before[2], col_buf_ptr);
}

template<typename T>
void ConvKernelGpuUtil<T>::NDHWCCol2Im(DeviceCtx* device_ctx, const T* col_buf_dptr,
                                       const Shape& in_shape, const Shape& weight_shape,
                                       const Shape& out_shape, const int32_t* strides,
                                       const int32_t* dilation_rate, const int32_t* padding_before,
                                       T* in_diff_ptr) {
  int32_t im_size = in_shape.Count(1);
  NDHWCCol2ImGpu<T>
      <<<BlocksNum4ThreadsNum(im_size), kCudaThreadsNumPerBlock, 0, device_ctx->cuda_stream()>>>(
          im_size, col_buf_dptr, in_shape.At(1), in_shape.At(2), in_shape.At(3), in_shape.At(4),
          weight_shape.At(2), weight_shape.At(3), weight_shape.At(4), out_shape.At(2),
          out_shape.At(3), out_shape.At(4), strides[0], strides[1], strides[2], dilation_rate[0],
          dilation_rate[1], dilation_rate[2], padding_before[0], padding_before[1],
          padding_before[2], in_diff_ptr);
}

#define INSTANTIATE_CONV_KERNEL(type_cpp, type_proto) \
  template class ConvKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_KERNEL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
