#include "oneflow/core/kernel/convolution_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

inline bool IsAGreaterThanZeroAndLessThanB(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

}  // namespace

template<typename T>
class ConvolutionKernelUtil<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionKernelUtil);
  ConvolutionKernelUtil() = delete;

  static void Col2Im(const KernelCtx& ctx, const T* data_col,
                     const int channels, const int height, const int width,
                     const int kernel_h, const int kernel_w, const int pad_h,
                     const int pad_w, const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w, T* mut_dptr) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() mutable {
      memset(mut_dptr, 0, height * width * channels * sizeof(T));
      const int output_h =
          (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h
          + 1;
      const int output_w =
          (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w
          + 1;
      const int channel_size = height * width;
      for (int channel = channels; channel--; mut_dptr += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {
          for (int kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {
            int input_row = -pad_h + kernel_row * dilation_h;
            for (int output_rows = output_h; output_rows; --output_rows) {
              if (!IsAGreaterThanZeroAndLessThanB(input_row, height)) {
                data_col += output_w;
              } else {
                int input_col = -pad_w + kernel_col * dilation_w;
                for (int output_col = output_w; output_col; --output_col) {
                  if (IsAGreaterThanZeroAndLessThanB(input_col, width)) {
                    mut_dptr[input_row * width + input_col] += *data_col;
                  }
                  ++data_col;
                  input_col += stride_w;
                }
              }
              input_row += stride_h;
            }
          }
        }
      }
    });
  }

  static void Im2Col(const KernelCtx& ctx, const T* dptr, const int channels,
                     const int height, const int width, const int kernel_h,
                     const int kernel_w, const int pad_h, const int pad_w,
                     const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w, T* data_col) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() mutable {
      const int output_h =
          (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h
          + 1;
      const int output_w =
          (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w
          + 1;
      const int channel_size = height * width;
      for (int channel = channels; channel--; dptr += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {
          for (int kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {
            int input_row = -pad_h + kernel_row * dilation_h;
            for (int output_rows = output_h; output_rows; --output_rows) {
              if (!IsAGreaterThanZeroAndLessThanB(input_row, height)) {
                for (int output_cols = output_w; output_cols; --output_cols) {
                  *(data_col++) = 0;
                }
              } else {
                int input_col = -pad_w + kernel_col * dilation_w;
                for (int output_col = output_w; output_col; --output_col) {
                  if (IsAGreaterThanZeroAndLessThanB(input_col, width)) {
                    *(data_col++) = dptr[input_row * width + input_col];
                  } else {
                    *(data_col++) = 0;
                  }
                  input_col += stride_w;
                }
              }
              input_row += stride_h;
            }
          }
        }
      }
    });
  }
};

template<DeviceType device_type, typename T>
ConvolutionKernel<device_type, T>::ConvolutionKernel() {
#ifdef USE_CUDNN
  CudaCheck(cudnnCreateTensorDescriptor(&in_desc_));
  CudaCheck(cudnnCreateTensorDescriptor(&out_desc_));
  CudaCheck(cudnnCreateFilterDescriptor(&weight_desc_));
  CudaCheck(cudnnCreateConvolutionDescriptor(&conv_desc_));
  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    CudaCheck(cudnnCreateTensorDescriptor(&bias_desc_));
  }
#endif  // USE_CUDNN
}

template<DeviceType device_type, typename T>
ConvolutionKernel<device_type, T>::~ConvolutionKernel() {
#ifdef USE_CUDNN
  CudaCheck(cudnnDestroyTensorDescriptor(in_desc_));
  CudaCheck(cudnnDestroyTensorDescriptor(out_desc_));
  CudaCheck(cudnnDestroyFilterDescriptor(weight_desc_));
  CudaCheck(cudnnDestroyConvolutionDescriptor(conv_desc_));
  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    CudaCheck(cudnnDestroyTensorDescriptor(bias_desc_));
  }
#endif  // USE_CUDNN
}

template<DeviceType device_type, typename T>
void ConvolutionKernel<device_type, T>::InitFromOpProto(
    const OperatorProto& op_proto) {
#ifdef USE_CUDNN
  Kernel::InitFromOpProto(op_proto);
  const auto conv_conf = op()->op_conf().convolution_conf();
  CudaCheck(cudnnSetConvolution2dDescriptor(
      conv_desc_, conv_conf.pad_h(), conv_conf.pad_w(), conv_conf.stride_h(),
      conv_conf.stride_w(), 1, 1, CUDNN_CROSS_CORRELATION));
#endif
}

template<DeviceType device_type, typename T>
void ConvolutionKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Shape& in_shape = in_blob->shape();
  Blob* out_blob = BnInOp2Blob("out");
  const Blob* weight_blob = BnInOp2Blob("weight");
  auto conv_conf = op()->op_conf().convolution_conf();
  CopyDataIdFromSoleIbToAllObIfNeed<device_type>(ctx, BnInOp2Blob);

#ifdef USE_CUDNN
  CudaCheck(cudnnSetTensor4dDescriptor(
      in_desc_, CUDNN_TENSOR_NCHW, cudnn::DataType<T>::type,
      in_blob->shape().At(0), in_blob->shape().At(1), in_blob->shape().At(2),
      in_blob->shape().At(3)));
  CudaCheck(cudnnSetTensor4dDescriptor(
      out_desc_, CUDNN_TENSOR_NCHW, cudnn::DataType<T>::type,
      out_blob->shape().At(0), out_blob->shape().At(1), out_blob->shape().At(2),
      out_blob->shape().At(3)));
  CudaCheck(cudnnSetFilter4dDescriptor(
      weight_desc_, cudnn::DataType<T>::type, CUDNN_TENSOR_NCHW,
      weight_blob->shape().At(0), weight_blob->shape().At(1),
      weight_blob->shape().At(2), weight_blob->shape().At(3)));

  cudnnConvolutionFwdAlgo_t cudnn_fwd_algo =
      (cudnnConvolutionFwdAlgo_t)(conv_conf.cudnn_fwd_algo());

  Blob* fwd_workspace = BnInOp2Blob("fwd_workspace");

  CudaCheck(cudnnConvolutionForward(
      ctx.device_ctx->cudnn_handle(), cudnn::DataType<T>::one, in_desc_,
      in_blob->dptr<T>(), weight_desc_, weight_blob->dptr<T>(), conv_desc_,
      cudnn_fwd_algo, fwd_workspace->mut_dptr<T>(),
      fwd_workspace->shape().At(0), cudnn::DataType<T>::zero, out_desc_,
      out_blob->mut_dptr<T>()));

  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    CudaCheck(cudnnSetTensor4dDescriptor(bias_desc_, CUDNN_TENSOR_NCHW,
                                         cudnn::DataType<T>::type, 1,
                                         out_blob->shape().At(1), 1, 1));
    const Blob* bias_blob = BnInOp2Blob("bias");
    CudaCheck(cudnnAddTensor(ctx.device_ctx->cudnn_handle(),
                             cudnn::DataType<T>::one, bias_desc_,
                             bias_blob->dptr<T>(), cudnn::DataType<T>::one,
                             out_desc_, out_blob->mut_dptr<T>()));
  }
#else
  Blob* col_buf_blob = BnInOp2Blob("col_buf");
  const int64_t in_im_sz = in_shape.Count(1);
  const int64_t out_im_sz = out_blob->shape().Count(1);
  const int64_t col_im_sz = col_buf_blob->shape().Count(1);
  for (size_t i = 0; i < in_shape.At(0); ++i) {
    ConvolutionKernelUtil<device_type, T>::Im2Col(
        ctx, in_blob->dptr<T>() + i * in_im_sz, in_shape.At(1), in_shape.At(2),
        in_shape.At(3), conv_conf.kernel_h(), conv_conf.kernel_w(),
        conv_conf.pad_h(), conv_conf.pad_w(), conv_conf.stride_h(),
        conv_conf.stride_w(), conv_conf.dilation_h(), conv_conf.dilation_w(),
        col_buf_blob->mut_dptr<T>() + i * col_im_sz);

    KernelUtil<device_type, T>::BlasGemm(
        ctx.device_ctx, CBLAS_ORDER::CblasRowMajor, CblasNoTrans, CblasTrans,
        out_blob->shape().At(1), out_blob->shape().Count(2),
        weight_blob->shape().At(1), static_cast<T>(1.0), weight_blob->dptr<T>(),
        weight_blob->shape().At(1), col_buf_blob->dptr<T>() + i * col_im_sz,
        weight_blob->shape().At(1), static_cast<T>(0.0),
        out_blob->mut_dptr<T>() + i * out_im_sz, col_buf_blob->shape().At(1));

    if (op()->GetBoolFromSpecialConf("has_bias_term")) {
      const Blob* bias_blob = BnInOp2Blob("bias");
      const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");

      // out_data = bias * bias_multiplier + out_data
      KernelUtil<device_type, T>::BlasGemm(
          ctx.device_ctx, CBLAS_ORDER::CblasRowMajor, CblasNoTrans,
          CblasNoTrans, bias_blob->shape().At(0), bias_mul_blob->shape().At(0),
          1, static_cast<T>(1.0), bias_blob->dptr<T>(), 1,
          bias_mul_blob->dptr<T>(), bias_mul_blob->shape().At(0),
          static_cast<T>(1.0), out_blob->mut_dptr<T>() + i * out_im_sz,
          bias_mul_blob->shape().At(0));
    }
  }
#endif  // USE_CUDNN
}

template<DeviceType device_type, typename T>
void ConvolutionKernel<device_type, T>::ComputeWeightDiff(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  auto conv_conf = op()->op_conf().convolution_conf();

#ifdef USE_CUDNN
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_blob = BnInOp2Blob("out");
  Blob* bwd_weight_workspace = BnInOp2Blob("bwd_weight_workspace");

  cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_weight_algo =
      (cudnnConvolutionBwdFilterAlgo_t)(conv_conf.cudnn_bwd_weight_algo());

  CudaCheck(cudnnConvolutionBackwardFilter(
      ctx.device_ctx->cudnn_handle(), cudnn::DataType<T>::one, in_desc_,
      in_blob->dptr<T>(), out_desc_, out_blob->dptr<T>(), conv_desc_,
      cudnn_bwd_weight_algo, bwd_weight_workspace->mut_dptr<T>(),
      bwd_weight_workspace->shape().At(0), cudnn::DataType<T>::one,
      weight_desc_, weight_diff_blob->mut_dptr<T>()));
#else
  const Blob* col_buf_blob = BnInOp2Blob("col_buf");
  const int64_t out_im_sz = out_diff_blob->shape().Count(1);
  const int64_t data_num = out_diff_blob->shape().At(0);
  const int64_t conv_sliding_window_steps = out_diff_blob->shape().Count(2);

  Memset<device_type>(ctx.device_ctx, weight_diff_blob->mut_dptr(), 0,
                      weight_diff_blob->ByteSizeOfDataField());
  for (size_t i = 0; i < data_num; ++i) {
    KernelUtil<device_type, T>::BlasGemm(
        ctx.device_ctx, CBLAS_ORDER::CblasRowMajor, CblasNoTrans, CblasNoTrans,
        weight_diff_blob->shape().At(0), weight_diff_blob->shape().At(1),
        out_diff_blob->shape().Count(2),
        static_cast<T>(1.0) / conv_sliding_window_steps,
        out_diff_blob->dptr<T>() + i * out_im_sz,
        out_diff_blob->shape().Count(2),
        col_buf_blob->dptr<T>() + i * col_buf_blob->shape().Count(1),
        col_buf_blob->shape().At(2), static_cast<T>(1.0),
        weight_diff_blob->mut_dptr<T>(), weight_diff_blob->shape().At(1));
  }
#endif  // USE_CUDNN
}

template<DeviceType device_type, typename T>
void ConvolutionKernel<device_type, T>::ComputeBiasDiff(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* bias_diff_blob = BnInOp2Blob("bias_diff");

#ifdef USE_CUDNN
  CudaCheck(cudnnConvolutionBackwardBias(
      ctx.device_ctx->cudnn_handle(), cudnn::DataType<T>::one, out_desc_,
      out_diff_blob->dptr<T>(), cudnn::DataType<T>::one, bias_desc_,
      bias_diff_blob->mut_dptr<T>()));
#else
  const int64_t out_im_sz = out_diff_blob->shape().Count(1);
  const int64_t data_num = out_diff_blob->shape().At(0);
  const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");
  const int64_t conv_sliding_window_steps = out_diff_blob->shape().Count(2);

  Memset<device_type>(ctx.device_ctx, bias_diff_blob->mut_dptr(), 0,
                      bias_diff_blob->ByteSizeOfDataField());
  for (size_t i = 0; i < data_num; ++i) {
    KernelUtil<device_type, T>::BlasGemm(
        ctx.device_ctx, CBLAS_ORDER::CblasRowMajor, CblasNoTrans, CblasNoTrans,
        bias_diff_blob->shape().At(0), 1, bias_mul_blob->shape().At(0),
        static_cast<T>(1.0) / conv_sliding_window_steps,
        out_diff_blob->dptr<T>() + i * out_im_sz,
        out_diff_blob->shape().Count(2), bias_mul_blob->dptr<T>(), 1,
        static_cast<T>(1.0), bias_diff_blob->mut_dptr<T>(), 1);
  }
#endif  // USE_CUDNN
}

template<DeviceType device_type, typename T>
void ConvolutionKernel<device_type, T>::ComputeInputDiff(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }

  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* weight_blob = BnInOp2Blob("weight");
  auto conv_conf = op()->op_conf().convolution_conf();

#ifdef USE_CUDNN
  Blob* bwd_data_workspace = BnInOp2Blob("bwd_data_workspace");
  cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo =
      (cudnnConvolutionBwdDataAlgo_t)(conv_conf.cudnn_bwd_data_algo());

  CudaCheck(cudnnConvolutionBackwardData(
      ctx.device_ctx->cudnn_handle(), cudnn::DataType<T>::one, weight_desc_,
      weight_blob->dptr<T>(), out_desc_, out_diff_blob->dptr<T>(), conv_desc_,
      cudnn_bwd_data_algo, bwd_data_workspace->mut_dptr<T>(),
      bwd_data_workspace->shape().At(0), cudnn::DataType<T>::zero, in_desc_,
      in_diff_blob->mut_dptr<T>()));
#else
  Blob* col_buf_blob = BnInOp2Blob("col_buf");

  const int64_t out_im_sz = out_diff_blob->shape().Count(1);
  const int64_t data_num = out_diff_blob->shape().At(0);
  for (size_t i = 0; i < data_num; ++i) {
    KernelUtil<device_type, T>::BlasGemm(
        ctx.device_ctx, CBLAS_ORDER::CblasRowMajor, CblasTrans, CblasNoTrans,
        col_buf_blob->shape().At(1), col_buf_blob->shape().At(2),
        weight_blob->shape().At(0), static_cast<T>(1.0),
        out_diff_blob->dptr<T>() + i * out_im_sz,
        out_diff_blob->shape().Count(2), weight_blob->dptr<T>(),
        weight_blob->shape().At(1), static_cast<T>(0.0),
        col_buf_blob->mut_dptr<T>() + i * col_buf_blob->shape().Count(1),
        col_buf_blob->shape().At(2));
  }

  const Shape& in_diff_shape = in_diff_blob->shape();
  const ConvolutionOpConf& conv_conf = op()->op_conf().convolution_conf();
  for (size_t i = 0; i < data_num; ++i) {
    ConvolutionKernelUtil<device_type, T>::Col2Im(
        ctx, col_buf_blob->dptr<T>() + i * col_buf_blob->shape().Count(1),
        in_diff_shape.At(1), in_diff_shape.At(2), in_diff_shape.At(3),
        conv_conf.kernel_h(), conv_conf.kernel_w(), conv_conf.pad_h(),
        conv_conf.pad_w(), conv_conf.stride_h(), conv_conf.stride_w(),
        conv_conf.dilation_h(), conv_conf.dilation_w(),
        in_diff_blob->mut_dptr<T>() + i * in_diff_shape.Count(1));
  }
#endif  // USE_CUDNN
}

template<DeviceType device_type, typename T>
void ConvolutionKernel<device_type, T>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ComputeWeightDiff(ctx, BnInOp2Blob);
  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    ComputeBiasDiff(ctx, BnInOp2Blob);
  }
  ComputeInputDiff(ctx, BnInOp2Blob);
}

template<DeviceType device_type, typename T>
void ConvolutionKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::FillWithProperConf(
      ctx.device_ctx,
      OF_PB_POINTER_GET(op()->op_conf().convolution_conf(), weight_fill),
      random_seed_gen(), BnInOp2Blob("weight"));

  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    KernelUtil<device_type, T>::FillWithProperConf(
        ctx.device_ctx,
        OF_PB_POINTER_GET(op()->op_conf().convolution_conf(), bias_fill),
        random_seed_gen(), BnInOp2Blob("bias"));
  }
}

template<DeviceType device_type, typename T>
void ConvolutionKernel<device_type, T>::InitModelBlobsWithDir(
    const KernelCtx& ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* weight_blob = BnInOp2Blob("weight");
  int32_t dim_num = op()->GetInt32FromSpecialConf("out_num");
  KernelUtil<device_type, T>::FillWithModelDir(
      ctx.device_ctx, part_id, part_num, model_load_dir, weight_blob, "weight",
      dim_num, weight_blob->shape().Count(1));
  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    KernelUtil<device_type, T>::FillWithModelDir(
        ctx.device_ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias"),
        "bias", dim_num, 1);
  }
}

template<DeviceType device_type, typename T>
void ConvolutionKernel<device_type, T>::InitModelTmpBlobs(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    FillConf bias_multiplier_fill_conf;
    bias_multiplier_fill_conf.mutable_constant_conf()->set_value(1.0f);
    KernelUtil<device_type, T>::Fill(ctx.device_ctx, bias_multiplier_fill_conf,
                                     0, BnInOp2Blob("bias_multiplier"));
  }
}

namespace {

Kernel* CreateConvolutionKenrel(const OpContext& op_ctx) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define CONVOLUTION_KERNEL_ENTRY(device_type, data_type_pair)          \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(data_type_pair)), []() {  \
     return new ConvolutionKernel<device_type,                         \
                                  OF_PP_PAIR_FIRST(data_type_pair)>(); \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
          CONVOLUTION_KERNEL_ENTRY, DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)};
  return creators.at(
      GetHashKey(op_ctx.device_type(), op_ctx.bn_in_op2data_type().at("in")))();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kConvolutionConf,
                         CreateConvolutionKenrel))

}  // namespace oneflow
