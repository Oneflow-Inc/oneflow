#include "oneflow/core/kernel/conv_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void ConvKernel<DeviceType::kGPU, float16>::VirtualKernelInit() {
  CHECK(this->EnableCudnn());
  Shape in_shape(this->GetConvKernelConf().in());
  Shape out_shape(this->GetConvKernelConf().out());
  Shape weight_shape(this->GetConvKernelConf().weight());

  const std::string& data_format =
      this->template GetValFromCustomizedOpConf<std::string>("data_format");
  this->in_desc_.reset(new CudnnTensorDesc(GetDataType<float16>::value, in_shape, data_format));
  this->out_desc_.reset(new CudnnTensorDesc(GetDataType<float16>::value, out_shape, data_format));
  this->filter_desc_.reset(
      new CudnnFilterDesc(GetDataType<float16>::value, weight_shape, data_format));
  this->conv_desc_.reset(new CudnnConvDesc(GetConvDescDataType(GetDataType<float16>::value),
                                           in_shape, this->GetCustomizedOpConf()));

  if (this->template GetValFromCustomizedOpConf<bool>("use_bias")) {
    int32_t filters = Shape(this->GetConvKernelConf().bias()).At(0);
    if ((this->OpKernelDim() == 1) || (this->OpKernelDim() == 2)) {
      if (data_format == "channels_first") {
        this->bias_desc_.reset(
            new CudnnTensorDesc(CUDNN_TENSOR_NCHW, GetDataType<float16>::value, 1, filters, 1, 1));
      } else if (data_format == "channels_last") {
        if (GetDataType<float16>::value == DataType::kDouble) {
          LOG(FATAL) << "CUDNN 1d & 2d support channels last only if data type "
                        "is float";
        }
        this->bias_desc_.reset(
            new CudnnTensorDesc(CUDNN_TENSOR_NHWC, GetDataType<float16>::value, 1, filters, 1, 1));
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
      this->bias_desc_.reset(new CudnnTensorDesc(GetDataType<float16>::value,
                                                 this->OpKernelDim() + 2, bias_dim.data(),
                                                 stride_of_bias_tensor.data()));
    }
  }
}

void ConvKernel<DeviceType::kGPU, float16>::DoForwardDataContent(
    DeviceCtx* device_ctx, const Blob* in_blob, const Blob* weight_blob, Blob* out_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(this->EnableCudnn());
  Blob* fw_cudnn_buf = BnInOp2Blob("fw_cudnn_buf");
  void* fw_cudnn_buf_ptr = fw_cudnn_buf ? fw_cudnn_buf->mut_dptr() : nullptr;
  size_t fw_cudnn_buf_size = fw_cudnn_buf ? fw_cudnn_buf->ByteSizeOfDataContentField() : 0;
  CudaCheck(cudnnConvolutionForward(
      device_ctx->cudnn_handle(), CudnnSPOnePtr<float16>(), this->in_desc_->Get(),
      in_blob->dptr<float16>(), this->filter_desc_->Get(), weight_blob->dptr<float16>(),
      this->conv_desc_->Get(),
      static_cast<cudnnConvolutionFwdAlgo_t>(this->GetConvKernelConf().cudnn_fwd_algo()),
      fw_cudnn_buf_ptr, fw_cudnn_buf_size, CudnnSPZeroPtr<float16>(), this->out_desc_->Get(),
      out_blob->mut_dptr<float16>()));

  if (this->template GetValFromCustomizedOpConf<bool>("use_bias")) {
    const Blob* bias = BnInOp2Blob("bias");
    CudaCheck(cudnnAddTensor(device_ctx->cudnn_handle(), CudnnSPOnePtr<float16>(),
                             this->bias_desc_->Get(), bias->dptr<float16>(),
                             CudnnSPOnePtr<float16>(), this->out_desc_->Get(),
                             out_blob->mut_dptr<float16>()));
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::VirtualKernelInit() {
  if (this->EnableCudnn()) {
    KernelInitWithCudnn();
  } else {
    ConvKernelImplByIm2Col<DeviceType::kGPU, T>::VirtualKernelInit();
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::DoForwardDataContent(
    DeviceCtx* device_ctx, const Blob* in_blob, const Blob* weight_blob, Blob* out_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->EnableCudnn()) {
    DoForwardDataContentWithCudnn(device_ctx, in_blob, weight_blob, out_blob, BnInOp2Blob);
  } else {
    ConvKernelImplByIm2Col<DeviceType::kGPU, T>::DoForwardDataContent(
        device_ctx, in_blob, weight_blob, out_blob, BnInOp2Blob);
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::WeightBackward(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, const Blob* in_blob, Blob* weight_diff_blob,
    Blob* in_diff_blob, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->EnableCudnn()) {
    WeightBackwardWithCudnn(device_ctx, out_diff_blob, in_blob, weight_diff_blob, in_diff_blob,
                            BnInOp2Blob);
  } else {
    ConvKernelImplByIm2Col<DeviceType::kGPU, T>::WeightBackward(
        device_ctx, out_diff_blob, in_blob, weight_diff_blob, in_diff_blob, BnInOp2Blob);
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::BiasBackward(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, Blob* bias_diff_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(this->op_conf().trainable());
  if (this->EnableCudnn()) {
    BiasBackwardWithCudnn(device_ctx, out_diff_blob, bias_diff_blob, BnInOp2Blob);
  } else {
    ConvKernelImplByIm2Col<DeviceType::kGPU, T>::BiasBackward(device_ctx, out_diff_blob,
                                                              bias_diff_blob, BnInOp2Blob);
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::KernelInitWithCudnn() {
  Shape in_shape(this->GetConvKernelConf().in());
  Shape out_shape(this->GetConvKernelConf().out());
  Shape weight_shape(this->GetConvKernelConf().weight());

  const std::string& data_format =
      this->template GetValFromCustomizedOpConf<std::string>("data_format");
  this->in_desc_.reset(new CudnnTensorDesc(GetDataType<T>::value, in_shape, data_format));
  this->out_desc_.reset(new CudnnTensorDesc(GetDataType<T>::value, out_shape, data_format));
  this->filter_desc_.reset(new CudnnFilterDesc(GetDataType<T>::value, weight_shape, data_format));
  this->conv_desc_.reset(new CudnnConvDesc(GetConvDescDataType(GetDataType<T>::value), in_shape,
                                           this->GetCustomizedOpConf()));

  if (this->template GetValFromCustomizedOpConf<bool>("use_bias")) {
    int32_t filters = Shape(this->GetConvKernelConf().bias()).At(0);
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
  Blob* fw_cudnn_buf = BnInOp2Blob("fw_cudnn_buf");
  void* fw_cudnn_buf_ptr = fw_cudnn_buf ? fw_cudnn_buf->mut_dptr() : nullptr;
  size_t fw_cudnn_buf_size = fw_cudnn_buf ? fw_cudnn_buf->ByteSizeOfDataContentField() : 0;
  CudaCheck(cudnnConvolutionForward(
      device_ctx->cudnn_handle(), CudnnSPOnePtr<T>(), this->in_desc_->Get(), in_blob->dptr<T>(),
      this->filter_desc_->Get(), weight_blob->dptr<T>(), this->conv_desc_->Get(),
      static_cast<cudnnConvolutionFwdAlgo_t>(this->GetConvKernelConf().cudnn_fwd_algo()),
      fw_cudnn_buf_ptr, fw_cudnn_buf_size, CudnnSPZeroPtr<T>(), this->out_desc_->Get(),
      out_blob->mut_dptr<T>()));

  if (this->template GetValFromCustomizedOpConf<bool>("use_bias")) {
    const Blob* bias = BnInOp2Blob("bias");
    CudaCheck(cudnnAddTensor(device_ctx->cudnn_handle(), CudnnSPOnePtr<T>(),
                             this->bias_desc_->Get(), bias->dptr<T>(), CudnnSPOnePtr<T>(),
                             this->out_desc_->Get(), out_blob->mut_dptr<T>()));
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::WeightBackwardWithCudnn(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, const Blob* in_blob, Blob* weight_diff_blob,
    Blob* in_diff_blob, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* bw_cudnn_buf = BnInOp2Blob("bw_cudnn_buf");
  void* bw_cudnn_buf_ptr = bw_cudnn_buf ? bw_cudnn_buf->mut_dptr() : nullptr;
  size_t bw_cudnn_buf_size = bw_cudnn_buf ? bw_cudnn_buf->ByteSizeOfDataContentField() : 0;
  if (this->op_conf().trainable()) {
    CudaCheck(cudnnConvolutionBackwardFilter(
        device_ctx->cudnn_handle(), CudnnSPOnePtr<T>(), this->in_desc_->Get(), in_blob->dptr<T>(),
        this->out_desc_->Get(), out_diff_blob->dptr<T>(), this->conv_desc_->Get(),
        static_cast<cudnnConvolutionBwdFilterAlgo_t>(
            this->GetConvKernelConf().cudnn_bwd_filter_algo()),
        bw_cudnn_buf_ptr, bw_cudnn_buf_size, CudnnSPZeroPtr<T>(), this->filter_desc_->Get(),
        weight_diff_blob->mut_dptr<T>()));
  }
  if (in_diff_blob != nullptr) {
    CudaCheck(cudnnConvolutionBackwardData(
        device_ctx->cudnn_handle(), CudnnSPOnePtr<T>(), this->filter_desc_->Get(),
        weight_blob->dptr<T>(), this->out_desc_->Get(), out_diff_blob->dptr<T>(),
        this->conv_desc_->Get(),
        static_cast<cudnnConvolutionBwdDataAlgo_t>(this->GetConvKernelConf().cudnn_bwd_data_algo()),
        bw_cudnn_buf_ptr, bw_cudnn_buf_size, CudnnSPZeroPtr<T>(), this->in_desc_->Get(),
        in_diff_blob->mut_dptr<T>()));
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::BiasBackwardWithCudnn(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, Blob* bias_diff_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CudaCheck(cudnnConvolutionBackwardBias(device_ctx->cudnn_handle(), CudnnSPOnePtr<T>(),
                                         this->out_desc_->Get(), out_diff_blob->dptr<T>(),
                                         CudnnSPZeroPtr<T>(), this->bias_desc_->Get(),
                                         bias_diff_blob->mut_dptr<T>()));
}

namespace {

template<int dim_num>
__device__ void InitSharedArrays(const int im_d, const int im_h, const int im_w, const int kernel_d,
                                 const int kernel_h, const int kernel_w, const int out_d,
                                 const int out_h, const int out_w, const int stride_d,
                                 const int stride_h, const int stride_w, const int dilation_d,
                                 const int dilation_h, const int dilation_w, const int pad_d,
                                 const int pad_h, const int pad_w, int* shared_im,
                                 int* shared_kernel, int* shared_out, int* shared_stride,
                                 int* shared_dilation, int* shared_pad) {
  if (threadIdx.x == 0) {
    if (dim_num == 3) {
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
    } else if (dim_num == 2) {
      shared_im[0] = im_h;
      shared_im[1] = im_w;
      shared_kernel[0] = kernel_h;
      shared_kernel[1] = kernel_w;
      shared_out[0] = out_h;
      shared_out[1] = out_w;
      shared_stride[0] = stride_h;
      shared_stride[1] = stride_w;
      shared_dilation[0] = dilation_h;
      shared_dilation[1] = dilation_w;
      shared_pad[0] = pad_h;
      shared_pad[1] = pad_w;
    } else if (dim_num == 1) {
      shared_im[0] = im_w;
      shared_kernel[0] = kernel_w;
      shared_out[0] = out_w;
      shared_stride[0] = stride_w;
      shared_dilation[0] = dilation_w;
      shared_pad[0] = pad_w;
    }
  }
  __syncthreads();
}

template<typename T, int dim_num, bool is_channel_first>
__global__ void Im2ColGpu(const int n, const T* im_dptr, const int channel, const int im_d,
                          const int im_h, const int im_w, const int kernel_d, const int kernel_h,
                          const int kernel_w, const int out_d, const int out_h, const int out_w,
                          const int stride_d, const int stride_h, const int stride_w,
                          const int dilation_rate_d, const int dilation_rate_h,
                          const int dilation_rate_w, const int padding_before_d,
                          const int padding_before_h, const int padding_before_w, T* col_buf_dptr) {
  __shared__ int shared_im[dim_num];
  __shared__ int shared_kernel[dim_num];
  __shared__ int shared_out[dim_num];
  __shared__ int shared_stride[dim_num];
  __shared__ int shared_dilation[dim_num];
  __shared__ int shared_pad[dim_num];
  InitSharedArrays<dim_num>(im_d, im_h, im_w, kernel_d, kernel_h, kernel_w, out_d, out_h, out_w,
                            stride_d, stride_h, stride_w, dilation_rate_d, dilation_rate_h,
                            dilation_rate_w, padding_before_d, padding_before_h, padding_before_w,
                            shared_im, shared_kernel, shared_out, shared_stride, shared_dilation,
                            shared_pad);

  int out_size = 1;
  for (int i = 0; i < dim_num; ++i) { out_size *= shared_out[i]; }
  int kernel_index[dim_num];
  int out_index[dim_num];
  int channel_index;
  int im_index[dim_num];
  CUDA_1D_KERNEL_LOOP(index, n) {
    // total launch channel*od*oh*ow threads,
    // each thread is responsible for a whole kernel size copy
    // calc kernel_/out_/channel_index
    channel_index = index / out_size;
    int col_offset = index % out_size;  // col_dim of col_buf: od*oh*ow
    for (int i = dim_num - 1; i >= 0; --i) {
      out_index[i] = col_offset % shared_out[i];
      col_offset /= shared_out[i];
      kernel_index[i] = 0;
    }

    int col_buf_offset = 0;
    if (is_channel_first) { col_buf_offset = channel_index; }
    for (int i = 0; i < 3; ++i) {
      col_buf_offset *= shared_kernel[i];
      // col_buf_offset += kernel_index[i]; commented for kernel_index[] == 0
    }
    if (is_channel_first == false) {
      col_buf_offset *= channel;
      col_buf_offset += channel_index;
    }
    col_buf_offset *= out_size;
    col_buf_offset += index % out_size;

    while (true) {
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
        int im_offset = 0;
        if (is_channel_first) { im_offset = channel_index; }
        for (int i = 0; i < dim_num; ++i) {
          im_offset *= shared_im[i];
          im_offset += im_index[i];
        }
        if (is_channel_first == false) {
          im_offset *= channel;
          im_offset += channel_index;
        }
        col_buf_dptr[col_buf_offset] = im_dptr[im_offset];
      } else {
        col_buf_dptr[col_buf_offset] = 0;
      }
      col_buf_offset += out_size;

      // loop over all kernel index
      bool is_loop_completed = true;
      for (int i = dim_num - 1; i >= 0; --i) {
        if (kernel_index[i] == shared_kernel[i] - 1) {
          kernel_index[i] = 0;
        } else {
          kernel_index[i] += 1;
          is_loop_completed = false;
          break;
        }
      }
      if (is_loop_completed) { break; }
    }
  }
}

template<typename T, int dim_num, bool is_channel_first>
__global__ void Col2ImGpu(const int n, const T* col_buf_dptr, const int channel, const int im_d,
                          const int im_h, const int im_w, const int kernel_d, const int kernel_h,
                          const int kernel_w, const int out_d, const int out_h, const int out_w,
                          const int stride_d, const int stride_h, const int stride_w,
                          const int dilation_rate_d, const int dilation_rate_h,
                          const int dilation_rate_w, const int padding_before_d,
                          const int padding_before_h, const int padding_before_w, T* im_diff_dptr) {
  __shared__ int shared_im[dim_num];
  __shared__ int shared_kernel[dim_num];
  __shared__ int shared_out[dim_num];
  __shared__ int shared_stride[dim_num];
  __shared__ int shared_dilation[dim_num];
  __shared__ int shared_pad[dim_num];
  InitSharedArrays<dim_num>(im_d, im_h, im_w, kernel_d, kernel_h, kernel_w, out_d, out_h, out_w,
                            stride_d, stride_h, stride_w, dilation_rate_d, dilation_rate_h,
                            dilation_rate_w, padding_before_d, padding_before_h, padding_before_w,
                            shared_im, shared_kernel, shared_out, shared_stride, shared_dilation,
                            shared_pad);

  int kernel_index[dim_num];
  int channel_index;
  int im_index[dim_num];
  int out_begin[dim_num];
  int out_end[dim_num];
  int out_index[dim_num];
  CUDA_1D_KERNEL_LOOP(index, n) {
    // calc im_/channel_index
    int im_offset = index;
    if (is_channel_first == false) {
      channel_index = im_offset % channel;
      im_offset /= channel;
    }
    for (int i = dim_num - 1; i >= 0; --i) {
      im_index[i] = im_offset % shared_im[i] + shared_pad[i];
      im_offset /= shared_im[i];
    }
    if (is_channel_first) { channel_index = im_offset; }

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
        if (is_channel_first) { col_buf_offset = channel_index; }
        for (int i = 0; i < dim_num; ++i) {
          col_buf_offset *= shared_kernel[i];
          col_buf_offset += kernel_index[i];
        }
        if (is_channel_first == false) {
          col_buf_offset *= channel;
          col_buf_offset += channel_index;
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

}  // namespace

#define IM2COL_KERNEL_CALL(kernel_func_name, dim_num, is_channel_first, kernel_num, src_dptr,     \
                           dst_dptr)                                                              \
  kernel_func_name<T, dim_num, is_channel_first>                                                  \
      <<<BlocksNum4ThreadsNum(kernel_num), kCudaThreadsNumPerBlock, 0,                            \
         device_ctx->cuda_stream()>>>(                                                            \
          kernel_num, src_dptr, in_shape.At(1), in_shape.At(2), in_shape.At(3), in_shape.At(4),   \
          weight_shape.At(2), weight_shape.At(3), weight_shape.At(4), out_shape.At(2),            \
          out_shape.At(3), out_shape.At(4), strides[0], strides[1], strides[2], dilation_rate[0], \
          dilation_rate[1], dilation_rate[2], padding_before[0], padding_before[1],               \
          padding_before[2], dst_dptr)

template<typename T>
void ConvKernelUtil<DeviceType::kGPU, T>::NCDHWIm2Col(
    const int dim_num, DeviceCtx* device_ctx, const T* in_dptr, const Shape& in_shape,
    const Shape& weight_shape, const Shape& out_shape, const int32_t* strides,
    const int32_t* dilation_rate, const int32_t* padding_before, T* col_buf_dptr) {
  int32_t kernels = weight_shape.At(1) * out_shape.Count(2);
  switch (dim_num) {
    case 1: IM2COL_KERNEL_CALL(Im2ColGpu, 1, true, kernels, in_dptr, col_buf_dptr); break;
    case 2: IM2COL_KERNEL_CALL(Im2ColGpu, 2, true, kernels, in_dptr, col_buf_dptr); break;
    case 3: IM2COL_KERNEL_CALL(Im2ColGpu, 3, true, kernels, in_dptr, col_buf_dptr); break;
    default: UNIMPLEMENTED();
  }
}

template<typename T>
void ConvKernelUtil<DeviceType::kGPU, T>::NDHWCIm2Col(
    const int dim_num, DeviceCtx* device_ctx, const T* in_dptr, const Shape& in_shape,
    const Shape& weight_shape, const Shape& out_shape, const int32_t* strides,
    const int32_t* dilation_rate, const int32_t* padding_before, T* col_buf_dptr) {
  int32_t kernels = weight_shape.At(1) * out_shape.Count(2);
  switch (dim_num) {
    case 1: IM2COL_KERNEL_CALL(Im2ColGpu, 1, false, kernels, in_dptr, col_buf_dptr); break;
    case 2: IM2COL_KERNEL_CALL(Im2ColGpu, 2, false, kernels, in_dptr, col_buf_dptr); break;
    case 3: IM2COL_KERNEL_CALL(Im2ColGpu, 3, false, kernels, in_dptr, col_buf_dptr); break;
    default: UNIMPLEMENTED();
  }
}

template<typename T>
void ConvKernelUtil<DeviceType::kGPU, T>::NCDHWCol2Im(
    const int dim_num, DeviceCtx* device_ctx, const T* col_buf_dptr, const Shape& in_shape,
    const Shape& weight_shape, const Shape& out_shape, const int32_t* strides,
    const int32_t* dilation_rate, const int32_t* padding_before, T* in_diff_dptr) {
  int32_t im_size = in_shape.Count(1);
  switch (dim_num) {
    case 1: IM2COL_KERNEL_CALL(Col2ImGpu, 1, true, im_size, col_buf_dptr, in_diff_dptr); break;
    case 2: IM2COL_KERNEL_CALL(Col2ImGpu, 2, true, im_size, col_buf_dptr, in_diff_dptr); break;
    case 3: IM2COL_KERNEL_CALL(Col2ImGpu, 3, true, im_size, col_buf_dptr, in_diff_dptr); break;
    default: UNIMPLEMENTED();
  }
}

template<typename T>
void ConvKernelUtil<DeviceType::kGPU, T>::NDHWCCol2Im(
    const int dim_num, DeviceCtx* device_ctx, const T* col_buf_dptr, const Shape& in_shape,
    const Shape& weight_shape, const Shape& out_shape, const int32_t* strides,
    const int32_t* dilation_rate, const int32_t* padding_before, T* in_diff_dptr) {
  int32_t im_size = in_shape.Count(1);
  switch (dim_num) {
    case 1: IM2COL_KERNEL_CALL(Col2ImGpu, 1, false, im_size, col_buf_dptr, in_diff_dptr); break;
    case 2: IM2COL_KERNEL_CALL(Col2ImGpu, 2, false, im_size, col_buf_dptr, in_diff_dptr); break;
    case 3: IM2COL_KERNEL_CALL(Col2ImGpu, 3, false, im_size, col_buf_dptr, in_diff_dptr); break;
    default: UNIMPLEMENTED();
  }
}

#undef IM2COL_KERNEL_CALL

#define INSTANTIATE_CONV_KERNEL(type_cpp, type_proto) \
  template class ConvKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_KERNEL, FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

#define INSTANTIATE_CONV_KERNEL_UTIL(type_cpp, type_proto) \
  template class ConvKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
