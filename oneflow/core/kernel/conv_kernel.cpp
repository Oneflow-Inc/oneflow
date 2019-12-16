#include "oneflow/core/kernel/conv_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* out_blob = BnInOp2Blob("out");
  DoForwardDataContent(ctx.device_ctx, in_blob, weight_blob, out_blob, BnInOp2Blob);
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->template GetValFromCustomizedOpConf<bool>("use_bias")
      && (device_type == DeviceType::kCPU || this->EnableCudnn() == false)) {
    InitializerConf bias_multiplier_initializer_conf;
    bias_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
    NewKernelUtil<device_type>::InitializeWithConstConf(
        ctx, bias_multiplier_initializer_conf.constant_conf(), BnInOp2Blob("bias_multiplier"));
  }
}

template<DeviceType device_type, typename T>
const PbMessage& ConvKernelIf<device_type, T>::GetCustomizedOpConf() const {
  CHECK(this->kernel_conf().has_conv_conf());
  switch (this->OpKernelDim()) {
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
const int32_t ConvKernelIf<device_type, T>::OpKernelDim() const {
  return this->GetConvKernelConf().dim();
}

template<DeviceType device_type, typename T>
void ConvKernelImplByIm2Col<device_type, T>::VirtualKernelInit() {
  const std::string& data_format =
      this->template GetValFromCustomizedOpConf<std::string>("data_format");
  if (data_format == "channels_first") {
    im2col_func_ = ConvKernelUtil<device_type, T>::NCDHWIm2Col;
    col2im_func_ = ConvKernelUtil<device_type, T>::NCDHWCol2Im;
    forward_func_ = KernelUtil<device_type, T>::OFGemm;
    dhw_offset_ = 2;
    is_out_diff_need_trans_ = CblasNoTrans;
  } else {
    im2col_func_ = ConvKernelUtil<device_type, T>::NDHWCIm2Col;
    col2im_func_ = ConvKernelUtil<device_type, T>::NDHWCCol2Im;
    forward_func_ = KernelUtil<device_type, T>::OFGemmTrans;
    dhw_offset_ = 1;
    is_out_diff_need_trans_ = CblasTrans;
  }
  in_shape_ = ShapeView(this->GetConvKernelConf().in());
  out_shape_ = ShapeView(this->GetConvKernelConf().out());
  weight_shape_ = ShapeView(this->GetConvKernelConf().weight());
  strides_ = this->GetConvKernelConf().strides().data();
  dilation_rate_ = this->GetConvKernelConf().dilation_rate().data();
  padding_before_ = this->GetConvKernelConf().pad_small_side().data();
}

template<DeviceType device_type, typename T>
void ConvKernelImplByIm2Col<device_type, T>::DoForwardDataContent(
    DeviceCtx* device_ctx, const Blob* in_blob, const Blob* weight_blob, Blob* out_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* fw_col_buf_blob = BnInOp2Blob("fw_col_buf");
  FOR_RANGE(int64_t, i, 0, in_shape_.At(0)) {
    im2col_func_(this->OpKernelDim(), device_ctx, GetImgDptr<T>(in_blob, i), in_shape_,
                 weight_shape_, out_shape_, strides_, dilation_rate_, padding_before_,
                 fw_col_buf_blob->mut_dptr<T>());

    // channels first: out = weight * col_buf
    // channels last:  out = (weight * col_buf)(T)
    forward_func_(device_ctx, CblasNoTrans, CblasNoTrans,
                  weight_shape_.At(0),                             // filter
                  out_shape_.Count(dhw_offset_, dhw_offset_ + 3),  // od * oh * ow
                  weight_shape_.Count(1),                          // ci * kd * kh * kw
                  static_cast<T>(1), weight_blob->dptr<T>(), fw_col_buf_blob->dptr<T>(),
                  static_cast<T>(0), GetImgMutDptr<T>(out_blob, i));

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

template<DeviceType device_type, typename T>
void ConvKernelImplByIm2Col<device_type, T>::WeightBackward(
    DeviceCtx* ctx, const Blob* out_diff_blob, const Blob* in_blob, Blob* weight_diff_blob,
    Blob* in_diff_blob, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* bw_col_buf_blob = BnInOp2Blob("bw_col_buf");
  Memset<device_type>(ctx, weight_diff_blob->mut_dptr<T>(), 0,
                      weight_diff_blob->ByteSizeOfBlobBody());
  if (in_diff_blob != nullptr) {
    Memset<device_type>(ctx, in_diff_blob->mut_dptr<T>(), 0, in_diff_blob->ByteSizeOfBlobBody());
  }
  FOR_RANGE(int64_t, i, 0, out_shape_.At(0)) {
    im2col_func_(this->OpKernelDim(), ctx, GetImgDptr<T>(in_blob, i), in_shape_, weight_shape_,
                 out_shape_, strides_, dilation_rate_, padding_before_,
                 bw_col_buf_blob->mut_dptr<T>());

    if (this->op_conf().trainable()) {
      // channels first:  weight' += out[i]' * col_buf(T)
      // channels last :  weight' += out[i]'(T) * col_buf(T)
      KernelUtil<device_type, T>::OFGemm(
          ctx, is_out_diff_need_trans_, CblasTrans,
          weight_shape_.At(0),                             //  filter
          weight_shape_.Count(1),                          //  ci * kd * kh * kw
          out_shape_.Count(dhw_offset_, dhw_offset_ + 3),  //  od * oh * ow
          static_cast<T>(1), GetImgDptr<T>(out_diff_blob, i), bw_col_buf_blob->dptr<T>(),
          static_cast<T>(1), weight_diff_blob->mut_dptr<T>());
    }

    if (in_diff_blob != nullptr) {
      // channels first:  col_buf' = weight(T) * out[i]'
      // channels last :  col_buf' = weight(T) * out[i]'(T)
      KernelUtil<device_type, T>::OFGemm(
          ctx, CblasTrans, is_out_diff_need_trans_,
          weight_shape_.Count(1),                          //  ci * kd * kh * kw
          out_shape_.Count(dhw_offset_, dhw_offset_ + 3),  //  od * oh * ow
          weight_shape_.At(0),                             //  filter
          static_cast<T>(1), weight_blob->dptr<T>(), GetImgDptr<T>(out_diff_blob, i),
          static_cast<T>(0), bw_col_buf_blob->mut_dptr<T>());

      // in' = col2im(col_buf')
      col2im_func_(this->OpKernelDim(), ctx, bw_col_buf_blob->dptr<T>(), in_shape_, weight_shape_,
                   out_shape_, strides_, dilation_rate_, padding_before_,
                   GetImgMutDptr<T>(in_diff_blob, i));
    }
  }
}

template<DeviceType device_type, typename T>
void ConvKernelImplByIm2Col<device_type, T>::BiasBackward(
    DeviceCtx* ctx, const Blob* out_diff_blob, Blob* bias_diff_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(this->op_conf().trainable());
  const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");
  Memset<device_type>(ctx, bias_diff_blob->mut_dptr<T>(), 0, bias_diff_blob->ByteSizeOfBlobBody());

  FOR_RANGE(int64_t, i, 0, out_shape_.At(0)) {
    // channels first:  bias' += out' * bias_mul
    // channels last:   bias' += out'(T) * bias_mul
    KernelUtil<device_type, T>::OFGemm(
        ctx, is_out_diff_need_trans_, CblasNoTrans,
        weight_shape_.At(0),                             //  filter
        1,                                               //  1
        out_shape_.Count(dhw_offset_, dhw_offset_ + 3),  //  od * oh * ow
        static_cast<T>(1), GetImgDptr<T>(out_diff_blob, i), bias_mul_blob->dptr<T>(),
        static_cast<T>(1), bias_diff_blob->mut_dptr<T>());
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv1DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv2DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv3DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);

template<typename T>
ColBufWriter<T>::ColBufWriter(const T* src_ptr, T* dst_ptr, int64_t c_size, int64_t id_size,
                              int64_t ih_size, int64_t iw_size, int64_t od_size, int64_t oh_size,
                              int64_t ow_size)
    : src_ptr_(src_ptr),
      dst_ptr_(dst_ptr),
      c_size_(c_size),
      id_size_(id_size),
      ih_size_(ih_size),
      iw_size_(iw_size),
      od_size_(od_size),
      oh_size_(oh_size),
      ow_size_(ow_size) {}

template<typename T>
Im2ColWriter<T>::Im2ColWriter(const T* src_ptr, T* dst_ptr, int64_t c_size, int64_t id_size,
                              int64_t ih_size, int64_t iw_size, int64_t od_size, int64_t oh_size,
                              int64_t ow_size)
    : ColBufWriter<T>::ColBufWriter(src_ptr, dst_ptr, c_size, id_size, ih_size, iw_size, od_size,
                                    oh_size, ow_size) {}

template<typename T>
void Im2ColWriter<T>::DHWCWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) {
  *(this->dst_ptr_++) =
      this->src_ptr_[id * this->id_size_ + ih * this->ih_size_ + iw * this->iw_size_ + c];
}

template<typename T>
void Im2ColWriter<T>::CDHWWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) {
  *(this->dst_ptr_++) = this->src_ptr_[id * this->id_size_ + ih * this->ih_size_ + iw];
}

template<typename T>
void Im2ColWriter<T>::InvalidDFunc() {
  FOR_RANGE(int64_t, i, 0, this->od_size_) { *(this->dst_ptr_++) = 0; }
}

template<typename T>
void Im2ColWriter<T>::InvalidHFunc() {
  FOR_RANGE(int64_t, i, 0, this->oh_size_) { *(this->dst_ptr_++) = 0; }
}

template<typename T>
void Im2ColWriter<T>::InvalidWFunc() {
  FOR_RANGE(int64_t, i, 0, this->ow_size_) { *(this->dst_ptr_++) = 0; }
}

template<typename T>
void Im2ColWriter<T>::NextImCSize() {
  this->src_ptr_ += this->c_size_;
}

template<typename T>
Col2ImWriter<T>::Col2ImWriter(const T* src_ptr, T* dst_ptr, int64_t c_size, int64_t id_size,
                              int64_t ih_size, int64_t iw_size, int64_t od_size, int64_t oh_size,
                              int64_t ow_size)
    : ColBufWriter<T>::ColBufWriter(src_ptr, dst_ptr, c_size, id_size, ih_size, iw_size, od_size,
                                    oh_size, ow_size) {}

template<typename T>
void Col2ImWriter<T>::DHWCWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) {
  this->dst_ptr_[id * this->id_size_ + ih * this->ih_size_ + iw * this->iw_size_ + c] +=
      *(this->src_ptr_++);
}

template<typename T>
void Col2ImWriter<T>::CDHWWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) {
  this->dst_ptr_[id * this->id_size_ + ih * this->ih_size_ + iw] += *(this->src_ptr_++);
}

template<typename T>
void Col2ImWriter<T>::InvalidDFunc() {
  this->src_ptr_ += this->od_size_;
}

template<typename T>
void Col2ImWriter<T>::InvalidHFunc() {
  this->src_ptr_ += this->oh_size_;
}

template<typename T>
void Col2ImWriter<T>::InvalidWFunc() {
  this->src_ptr_ += this->ow_size_;
}

template<typename T>
void Col2ImWriter<T>::NextImCSize() {
  this->dst_ptr_ += this->c_size_;
}

template<typename T>
ColBufUtil<T>::ColBufUtil(const ShapeView& in_shape, const ShapeView& out_shape, int32_t dhw_offset,
                          const int32_t* strides, const int32_t* dilation_rate,
                          const int32_t* padding_before)
    : strides_(strides), dilation_rate_(dilation_rate), padding_before_(padding_before) {
  id_num_ = in_shape.At(dhw_offset);
  ih_num_ = in_shape.At(dhw_offset + 1);
  iw_num_ = in_shape.At(dhw_offset + 2);
  od_num_ = out_shape.At(dhw_offset);
  oh_num_ = out_shape.At(dhw_offset + 1);
  ow_num_ = out_shape.At(dhw_offset + 2);
  if (dhw_offset == 2) {
    dhw_valid_func_ = &ColBufWriter<T>::CDHWWrite;
  } else {
    dhw_valid_func_ = &ColBufWriter<T>::DHWCWrite;
  }
}

template<typename T>
void ColBufUtil<T>::operator()(ColBufWriter<T>* col_buf_writer, int64_t c, int64_t kd, int64_t kh,
                               int64_t kw) {
  int64_t id = kd * dilation_rate_[0] - padding_before_[0];
  FOR_RANGE(int64_t, od, 0, od_num_) {
    if (id < 0 || id >= id_num_) {
      col_buf_writer->InvalidDFunc();
    } else {
      int64_t ih = kh * dilation_rate_[1] - padding_before_[1];
      FOR_RANGE(int64_t, oh, 0, oh_num_) {
        if (ih < 0 || ih >= ih_num_) {
          col_buf_writer->InvalidHFunc();
        } else {
          int64_t iw = kw * dilation_rate_[2] - padding_before_[2];
          FOR_RANGE(int64_t, ow, 0, ow_num_) {
            if (iw < 0 || iw >= iw_num_) {
              col_buf_writer->InvalidWFunc();
            } else {
              (col_buf_writer->*dhw_valid_func_)(c, id, ih, iw);
            }
            iw += strides_[2];
          }
        }
        ih += strides_[1];
      }
    }
    id += strides_[0];
  }
}

template<typename T>
void ConvKernelUtil<DeviceType::kCPU, T>::DoNCDWHFunc(const ShapeView& weight_shape,
                                                      ColBufUtil<T>& col_buf_util,
                                                      ColBufWriter<T>* col_buf_writer) {
  for (int64_t c = 0; c != weight_shape.At(1); col_buf_writer->NextImCSize(), ++c) {
    for (int64_t kd = 0; kd != weight_shape.At(2); ++kd) {
      for (int64_t kh = 0; kh != weight_shape.At(3); ++kh) {
        for (int64_t kw = 0; kw != weight_shape.At(4); ++kw) {
          col_buf_util(col_buf_writer, c, kd, kh, kw);
        }
      }
    }
  }
}

template<typename T>
void ConvKernelUtil<DeviceType::kCPU, T>::NCDHWIm2Col(
    const int dim_num, DeviceCtx* device_ctx, const T* in_dptr, const ShapeView& in_shape,
    const ShapeView& weight_shape, const ShapeView& out_shape, const int32_t* strides,
    const int32_t* dilation_rate, const int32_t* padding_before, T* col_buf_ptr) {
  ColBufUtil<T> col_buf_util(in_shape, out_shape, 2, strides, dilation_rate, padding_before);
  Im2ColWriter<T> col_buf_writer(in_dptr, col_buf_ptr, in_shape.Count(2), in_shape.Count(3),
                                 in_shape.Count(4), 1, out_shape.Count(3), out_shape.Count(4), 1);
  DoNCDWHFunc(weight_shape, col_buf_util, &col_buf_writer);
}

template<typename T>
void ConvKernelUtil<DeviceType::kCPU, T>::NCDHWCol2Im(
    const int dim_num, DeviceCtx* device_ctx, const T* col_buf_ptr, const ShapeView& in_shape,
    const ShapeView& weight_shape, const ShapeView& out_shape, const int32_t* strides,
    const int32_t* dilation_rate, const int32_t* padding_before, T* in_diff_ptr) {
  ColBufUtil<T> col_buf_util(in_shape, out_shape, 2, strides, dilation_rate, padding_before);
  Col2ImWriter<T> col_buf_writer(col_buf_ptr, in_diff_ptr, in_shape.Count(2), in_shape.Count(3),
                                 in_shape.Count(4), 1, out_shape.Count(3), out_shape.Count(4), 1);
  DoNCDWHFunc(weight_shape, col_buf_util, &col_buf_writer);
}

template<typename T>
void ConvKernelUtil<DeviceType::kCPU, T>::DoNDWHCFunc(const ShapeView& weight_shape,
                                                      ColBufUtil<T>& col_buf_util,
                                                      ColBufWriter<T>* col_buf_writer) {
  for (int64_t kd = 0; kd != weight_shape.At(1); ++kd) {
    for (int64_t kh = 0; kh != weight_shape.At(2); ++kh) {
      for (int64_t kw = 0; kw != weight_shape.At(3); ++kw) {
        for (int64_t c = 0; c != weight_shape.At(4); ++c) {
          col_buf_util(col_buf_writer, c, kd, kh, kw);
        }
      }
    }
  }
}

template<typename T>
void ConvKernelUtil<DeviceType::kCPU, T>::NDHWCIm2Col(
    const int dim_num, DeviceCtx* device_ctx, const T* in_dptr, const ShapeView& in_shape,
    const ShapeView& weight_shape, const ShapeView& out_shape, const int32_t* strides,
    const int32_t* dilation_rate, const int32_t* padding_before, T* col_buf_ptr) {
  ColBufUtil<T> col_buf_util(in_shape, out_shape, 1, strides, dilation_rate, padding_before);
  Im2ColWriter<T> col_buf_writer(in_dptr, col_buf_ptr, in_shape.Count(2), in_shape.Count(2),
                                 in_shape.Count(3), in_shape.Count(4), out_shape.Count(2, 4),
                                 out_shape.Count(3, 4), 1);
  DoNDWHCFunc(weight_shape, col_buf_util, &col_buf_writer);
}

template<typename T>
void ConvKernelUtil<DeviceType::kCPU, T>::NDHWCCol2Im(
    const int dim_num, DeviceCtx* device_ctx, const T* col_buf_ptr, const ShapeView& in_shape,
    const ShapeView& weight_shape, const ShapeView& out_shape, const int32_t* strides,
    const int32_t* dilation_rate, const int32_t* padding_before, T* in_diff_ptr) {
  ColBufUtil<T> col_buf_util(in_shape, out_shape, 1, strides, dilation_rate, padding_before);
  Col2ImWriter<T> col_buf_writer(col_buf_ptr, in_diff_ptr, in_shape.Count(2), in_shape.Count(2),
                                 in_shape.Count(3), in_shape.Count(4), out_shape.Count(2, 4),
                                 out_shape.Count(3, 4), 1);
  DoNDWHCFunc(weight_shape, col_buf_util, &col_buf_writer);
}

#define INSTANTIATE_CONV_KERNEL_IF(device_type, data_type_pair) \
  template class ConvKernelIf<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CONV_KERNEL_IF, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ);
template class ConvKernelIf<DeviceType::kGPU, float16>;

#define INSTANTIATE_CONV_KERNEL(type_cpp, type_proto) \
  template class ConvKernel<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_KERNEL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
