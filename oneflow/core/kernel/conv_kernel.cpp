#include "oneflow/core/kernel/conv_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
const T* img_offset(const Blob* blob, int idx) {
  return blob->dptr<T>() + blob->shape().At(0) * idx;
}

template<typename T>
T* mut_img_offset(Blob* blob, int idx) {
  return const_cast<T>(img_offset<T>(blob, idx));
}

}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  WeightForward(ctx.device_ctx, BnInOp2Blob);
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    BiasForward(ctx.device_ctx, BnInOp2Blob);
  }
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    BiasBackward(ctx.device_ctx, BnInOp2Blob);
  }
  WeightBackward(ctx.device_ctx, BnInOp2Blob);
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::InitPureModelTmpBlobs(
    DeviceCtx* ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    InitializerConf bias_multiplier_initializer_conf;
    bias_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
    KernelUtil<device_type, T>::Initialize(ctx,
                                           bias_multiplier_initializer_conf, 0,
                                           BnInOp2Blob("bias_multiplier"));
  }
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx, this->GetMessageFromCustomizedOpConf("weight_initializer"),
      (*random_seed_gen)(), BnInOp2Blob("weight"));

  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx, this->GetMessageFromCustomizedOpConf("bias_initializer"),
        (*random_seed_gen)(), BnInOp2Blob("bias"));
  }
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* weight_blob = BnInOp2Blob("weight");
  int32_t dim_num = this->GetInt32FromCustomizedOpConf("filters");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, weight_blob, "weight", dim_num,
      weight_blob->shape().Count(1));
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    KernelUtil<device_type, T>::InitializeWithModelDir(
        ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias"), "bias",
        dim_num, 1);
  }
}

template<DeviceType device_type, typename T>
const PbMessage& ConvKernelIf<device_type, T>::GetCustomizedOpConf() const {
  CHECK(this->kernel_conf().has_conv_conf());
  switch (KernelDim()) {
    case 1: return this->op_conf().conv_1d_conf();
    case 2: return this->op_conf().conv_2d_conf();
    case 3: return this->op_conf().conv_3d_conf();
    default: UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
const ConvKernelConf& ConvKernelIf<device_type, T>::GetConvKernelConf() const {
  return this->kernel_conf().conv_conf();
}

template<DeviceType device_type, typename T>
const int32_t ConvKernelIf<device_type, T>::KernelDim() const {
  return GetConvKernelConf().in().dim_size() - 2;
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::WeightForward(
    DeviceCtx* device_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* col_buf_blob = BnInOp2Blob("col_buf");
  FOR_RANGE(int64_t, i, 0, in_blob->shape().At(0)) {
    ConvKernelUtil<T>::Im2Col(
        device_ctx, img_offset(in_blob, i),
        in_blob->shape(), weight_blob->shape(), out_blob->shape(),
        this->op_conf().strides(), this->op_conf().dilation_rate(),
        this->kernel_conf().padding_before(), col_buf_blob->mut_dptr<T>());
    KernelUtil<DeviceType::kCPU, T>::Gemm(device_ctx, CblasRowMajor,
                                          CblasNoTrans, CblasNoTrans,
                                          out_blob->shape().At(1),
                                          out_blob->shape().Count(2),
                                          weight_blob->shape().Count(1),
                                          static_cast<T>(1), weight_blob->dptr<T>(),
                                          weight_blob->shape().Count(1),
                                          col_buf_blob->dptr<T>(),
                                          out_blob->shape().Count(2),
                                          static_cast<T>(0), img_offset(out_blob, i),
                                          out_blob->shape().Count(2));
  }
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::BiasForward(
    DeviceCtx* device_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* bias_blob = BnInOp2Blob("bias");
  Blob* out_blob = BnInOp2Blob("out");
  KernelUtil<DeviceType::kCPU, T>::Axpy(
      device_ctx, out_blob->shape().elem_cnt(), static_cast<T>(1),
      bias_blob->dptr<T>(), 1, out_blob->mut_dptr<T>(), 1);
  KernelUtil<DeviceType::kCPU, T>::BlobGemm(device_ctx, CblasNoTrans, CblasNoTrans,
                                            static_cast<T>(1), static_cast<T>(1),
                                            bias_blob, bias_mul_blob, out_blob);
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::WeightBackward(
    DeviceCtx* ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");
  Blob* col_buf_blob = BnInOp2Blob("col_buf");
  KernelUtil<DeviceType::kCPU, T>::Memset(ctx, weight_diff_blob->mut_dptr<T>(),
                                          0, weight_diff_blob->ByteSizeOfDataContentField());
  FOR_RANGE(int64_t, i, 0, in_diff_blob->shape().At(0)) {
    ConvKernelUtil<T>::Im2Col(
        ctx, in_blob->dptr<T>() + i * in_blob->shape().Count(1),
        in_blob->shape(), weight_diff_blob->shape(), out_diff_blob->shape(),
        this->op_conf().strides(), this->op_conf().dilation_rate(),
        this->kernel_conf().padding_before(), col_buf_blob->mut_dptr<T>());
    // weight' += out[i]' * col_buf
    KernelUtil<DeviceType::kCPU, T>::Gemm(ctx, CblasRowMajor, CblasNoTrans,
                                          CblasTrans, weight_diff_blob->shape().At(1),
                                          weight_diff_blob->shape().Count(2),
                                          out_diff_blob->shape().Count(2),
                                          static_cast<T>(1), img_offset(out_diff_blob, i),
                                          out_diff_blob->shape().Count(2),
                                          col_buf_blob->dptr<T>(),
                                          out_diff_blob->shape().Count(2),
                                          static_cast<T>(1),
                                          weight_diff_blob->mut_dptr<T>(),
                                          weight_diff_blob->shape().Count(2));
    // col_buf' = weight * out[i]'
    KernelUtil<DeviceType::kCPU, T>::Gemm(ctx, CblasRowMajor, CblasTrans, CblasNoTrans,
                                          col_buf_blob->shape().Count(0, 4),
                                          col_buf_blob->shape().Count(4),
                                          weight_blob->shape().At(0),
                                          static_cast<T>(1),
                                          weight_blob->dptr<T>(),
                                          col_buf_blob->shape().Count(0, 4),
                                          img_offset(out_diff_blob, i),
                                          col_buf_blob->shape().Count(4),
                                          static_cast<T>(0),
                                          col_buf_blob->mut_dptr<T>(),
                                          col_buf_blob->shape().Count(4));
  // col2im(col_buf')
  ConvKernelUtil<T>::Col2Im(ctx, col_buff_blob->dptr<T>(),
                            in_blob->shape(), weight_blob->shape(),
                            out_blob->shape(), this->op_conf().strides(),
                            this->op_conf().dilation_rate(),
                            this->kernel_conf().padding_before(), mut_img_offset(in_diff_blob, i));
  }
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::BiasBackward(
    DeviceCtx* ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* bias_diff_blob = BnInOp2Blob("bias_diff");
  const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), bias_mul_blob, out_diff_blob, bias_diff_blob);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv1DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv2DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv3DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);

template<typename T>
class ConvKernelUtil final {
 public:
  static void Im2Col(DeviceCtx* device_ctx, const T* in_dptr,
                     const Shape& in_shape, const Shape& weight_shape,
                     const Shape& out_shape, const PbRpf<int32_t>& strides,
                     const PbRpf<int32_t>& dilation_rate,
                     const PbRpf<int32_t>& padding_before, T* col_buf) {
    in_dptr += img_dix * in_shape.Count(1);
    int64_t id_size = in_shape.Count(3);
    int64_t ih_size = in_shape.Count(4);
    for (int64_t c = 0; c != in_shape.At(1); in += in_shape.Count(2)) {
      for (int64_t kd = 0; kd != weight_shape.At(1); ++kd) {
        for (int64_t kh = 0; kh != weight_shape.At(2); ++kh) {
          for (int64_t kw = 0; kw != weight_shape.At(3); ++kw) {
            int64_t id = kd * dilation_rate.Get(0) - padding_before.Get(0);
            for (int64_t od = out_shape.At(2); od > 0; od--) {
              if (id < 0 || id >= in_shape.At(2)) {
                FOR_RANGE(int64_t, out, 0, id_size) { *(col_buf++) = 0; }
              } else {
                int64_t ih = kh * dilation_rate.Get(1) - padding_before.Get(1);
                for (int64_t oh = out_shape.At(3); oh > 0; oh--) {
                  if (ih < 0 || ih >= in_shape.At(3)) {
                    FOR_RANGE(int64_t, out, 0, ih_size) { *(col_buf++) = 0; }
                  } else {
                    int64_t iw =
                        kw * dilation_rate.Get(2) - padding_before.Get(2);
                    for (int64_t ow = out_shape.At(4); ow > 0; ow--) {
                      if (iw < 0 || iw >= in_shape.At(4)) {
                        *(col_buf++) = 0;
                      } else {
                        *(col_buf++) =
                            in_dptr[id * id_size + ih * ih_size + iw];
                      }
                      iw += strides.Get(2);
                    }
                  }
                  ih += strides.Get(1);
                }
              }
              id += strides.Get(0);
            }  // od
          }    // kw
        }      // kh
      }        // kd
    }          // c
  }

  static void Col2Im(DeviceCtx* device_ctx, const T* col_buf,
                     const Shape& in_shape, const Shape& weight_shape,
                     const Shape& out_shape, const PbRpf<int32_t>& strides,
                     const PbRpf<int32_t>& dilation_rate,
                     const PbRpf<int32_t>& padding_before, T* in_diff_ptr) {
    int64_t id_size = in_shape.Count(3);
    int64_t ih_size = in_shape.Count(4);
    for (int64_t c = 0; c != weight_shape.At(1);
         in_diff_ptr += in_shape.Count(2)) {
      for (int64_t kd = 0; kd != weight_shape.At(1); ++kd) {
        for (int64_t kh = 0; kh != weight_shape.At(2); ++kh) {
          for (int64_t kw = 0; kw != weight_shape.At(3); ++kw) {
            int64_t id = kd * dilation_rate.Get(0) - padding_before.Get(0);
            for (int64_t od = 0; od != out_shape.At(2); ++od) {
              if (id < 0 || id >= in_shape.At(2)) {
                in_diff_ptr += id_size;
              } else {
                int64_t ih = kh * dilation_rate.Get(1) - padding_before.Get(1);
                for (int64_t oh = 0; oh != out_shape.At(3); ++oh) {
                  if (ih < 0 || ih >= in_shape.At(3)) {
                    in_diff_ptr += ih_size;
                  } else {
                    int64_t iw =
                        kw * dilation_rate.Get(1) - padding_before.Get(1);
                    for (int64_t ow = 0; ow != out_shape.At(4); ++ow) {
                      if (iw >= 0 && iw < in_shape.At(4)) {
                        in_diff_ptr[id * id_size + ih * ih_size + iw] +=
                            *col_buf;
                      }
                      col_buf++;
                      iw += strides.Get(2);
                    }
                  }
                  ih += strides.Get(1);
                }
              }
              id += strides.Get(0);
            }
          }
        }
      }
    }
  }
};

}  // namespace oneflow
