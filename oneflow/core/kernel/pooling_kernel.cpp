#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<typename FloatingPointType>
void PoolingKernelUtil<DeviceType::kCPU, FloatingPointType>::RangeMaxQuery(
    const FloatingPointType* in_dptr,
    const int64_t in_height, const int64_t in_width,
    const int64_t hstart, const int64_t wstart,
    const int64_t hend, const int64_t wend,
    const int64_t pool_size, const int64_t out_index,
    FloatingPointType* out_dptr, int64_t* mask_dptr) {
  out_dptr[out_index] = in_dptr[hstart * in_width + wstart];
  mask_dptr[out_index] = hstart * in_width + wstart;
  for (int64_t h = hstart; h < hend; ++h) {
    for (int64_t w = wstart; w < wend; ++w) {
      const int64_t index = h * in_width + w;
      if (in_dptr[index] > out_dptr[out_index]) {
        out_dptr[out_index] = in_dptr[index];
        mask_dptr[out_index] = index;
      }
    }
  }
}

template<typename FloatingPointType>
void PoolingKernelUtil<DeviceType::kCPU, FloatingPointType>::RangeAveQuery(
    const FloatingPointType* in_dptr,
    const int64_t in_height, const int64_t in_width,
    const int64_t hstart, const int64_t wstart,
    const int64_t hend, const int64_t wend,
    const int64_t pool_size, const int64_t out_index,
    FloatingPointType* out_dptr, int64_t* mask_dptr) {
  out_dptr[out_index] = 0;
  for (int64_t h = hstart; h < hend; ++h) {
    for (int64_t w = wstart; w < wend; ++w) {
      const int64_t index = h * in_width + w;
      out_dptr[out_index] += in_dptr[index];
    }
  }
  out_dptr[out_index] /= pool_size;
}

template<typename FloatingPointType>
void PoolingKernelUtil<DeviceType::kCPU, FloatingPointType>::RangeStoQuert(
    const FloatingPointType* in_dptr,
    const int64_t in_height, const int64_t in_width,
    const int64_t hstart, const int64_t wstart,
    const int64_t hend, const int64_t wend,
    const int64_t pool_size, const int64_t out_index,
    FloatingPointType* out_dptr, int64_t* mask_dptr) {
  const int64_t index = (hstart + random()) * in_width + (wstart + random());
  out_dptr[out_index] = in_dptr[index];
  mask_dptr[out_index] = index;
}

template<typename FloatingPointType>
void PoolingKernelUtil<DeviceType::kCPU, FloatingPointType>::PoolingMaxBp(
    const FloatingPointType* out_diff_dptr, const int64_t* mask_dptr,
    const int64_t pool_size, const int64_t out_diff_index,
    const int64_t in_height, const int64_t in_width,
    const int64_t hstart, const int64_t wstart,
    const int64_t hend, const int64_t wend,
    FloatingPointType* in_diff_dptr) {
  const int64_t in_diff_index = mask_dptr[out_diff_index];
  in_diff_dptr[in_diff_index] += out_diff_dptr[out_diff_index];
}

template<typename FloatingPointType>
void PoolingKernelUtil<DeviceType::kCPU, FloatingPointType>::PoolingAveBp(
    const FloatingPointType* out_diff_dptr, const int64_t* mask_dptr,
    const int64_t pool_size, const int64_t out_diff_index,
    const int64_t in_height, const int64_t in_width,
    const int64_t hstart, const int64_t wstart,
    const int64_t hend, const int64_t wend,
    FloatingPointType* in_diff_dptr) {
  for (int h = hstart; h < hend; ++h) {
    for (int w = wstart; w < wend; ++w) {
      in_diff_dptr[h * in_width + w] += out_diff_dptr[out_diff_index] / pool_size;
    }
  }
}

template<typename FloatingPointType>
void PoolingKernelUtil<DeviceType::kCPU, FloatingPointType>::PoolingStoBp(
    const FloatingPointType* out_diff_dptr, const int64_t* mask_dptr,
    const int64_t pool_size, const int64_t out_diff_index,
    const int64_t in_height, const int64_t in_width,
    const int64_t hstart, const int64_t wstart,
    const int64_t hend, const int64_t wend,
    FloatingPointType* in_diff_dptr) {
  const int64_t in_diff_index = mask_dptr[out_diff_index];
  in_diff_dptr[in_diff_index] += out_diff_dptr[out_diff_index];
}

template<DeviceType device_type, typename FloatingPointType>
void PoolingKernel::InitFromOpProto(const OperatorProto& op_proto) {
  Kernel::InitFromOpProto(op_proto);

  switch(op()->op_conf().pooling_conf().pool()) {
    case PoolingOpConf::MAX:
      PoolingMethodForwardFunc_ =
        &PoolingKernelUtil<device_type, FloatingPointType>RangeMaxQuery;
      PoolingMethodBackwardFunc_ =
        &PoolingKernelUtil<device_type, FloatingPointType>PoolingMaxBp;
      break;
    case PoolingOpConf::AVE:
      PoolingMethodForwardFunc_ =
        &PoolingKernelUtil<device_type, FloatingPointType>RangeAveQuery;
      PoolingMethodBackwardFunc_ =
        &PoolingKernelUtil<device_type, FloatingPointType>PoolingAveBp;
      break;
    case PoolingOpConf::STOCHASTIC:
      PoolingMethodForwardFunc_ =
        &PoolingKernelUtil<device_type, FloatingPointType>RangeStoQuery;
      PoolingMethodBackwardFunc_ =
        &PoolingKernelUtil<device_type, FloatingPointType>PoolingStoBp;
      break;
    default:
      UNEXPECTED_RUN();
  }
}

template<DeviceType device_type, typename FloatingPointType>
void PoolingKernel::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* mask_blob = BnInOp2Blob("idx");
  CHECK((in_blob->shape.axis() == 4) && (out_blob->shape.axis() == 4));
  const FloatingPointType* in_dptr =
    static_cast<const FloatingPointType*>(in_blob->dptr());
  FloatingPointType* out_dptr =
    static_cast<FloatingPointType*>(out_blob->mut_dptr());
  int64_t* mask_dptr =
    static_cast<int64_t*>(mask_blob->mut_dptr());

  for (int64_t n = 0; n < out_blob->shape().At(0); ++n) {
    for (int64_t c = 0; c < out_blob->shape().At(1); ++c) {
      for (int64_t out_h = 0; out_h < out_blob->shape().At(2); ++out_h) {
        for (int64_t out_w = 0; out_w < out_blob->shape().at(3); ++out_w) {
          int64_t hstart = out_h * stride_h_ - pad_h_;
          int64_t wstart = out_w * stride_w_ - pad_w_;
          int64_t hend = std::min(hstart + kernel_h_, in_blob->shape().At(2) + pad_h);
          int64_t wend = std::min(wstart + kernel_w_, in_blob->shape().At(3) + pad_w);
          int64_t pool_size = (hend - hstart) * (wend - wstart);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          hend = std::min(hend, in_blob->shape().At(2));
          wend = std::min(wend, in_blob->shape().At(3));
          const int64_t out_index = out_h * out_blob->shape().At(3) + out_w;
          PoolingMethodForwardFunc_(
              in_dptr, in_blob->shape().At(2), in_blob->shape().At(3), pool_size, 
              hstart, wstart, hend, wend, out_dptr, mask_dptr, out_index);
        }
      }
      in_dptr += in_blob->shape().At(2) * in_blob->shape().At(3);
      out_dptr += out_blob->shape().At(2) * out_blob->shape().At(3);
      mask_dptr += out_blob->shape().At(2) * out_blob->shape().At(3);
    }
  }
}

template<DeviceType device_type, typename FloatingPointType>
void PoolingKernel::Backward(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  if (BnInOp2Blob("in_diff") == nullptr) {
    return;
  }

  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* mask_blob = BnInOp2Blob("idx");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  CHECK((in_diff_blob->shape.axis() == 4) && (out_diff_blob->shape.axis() == 4));
  
  const FloatingPointType* out_diff_dptr =
    static_cast<const FloatingPointType*>(out_diff_blob->dptr());
  const int64_t* mask_dptr = static_cast<const int64_t*>(index_blob->dptr());
  FloatingPointType* in_diff_dptr =
    static_cast<FloatingPointType>(in_diff_blob->mut_dptr());

  for (int64_t n = 0; n < out_diff_blob->shape().At(0); ++n) {
    for (int64_t c = 0; c < out_diff_blob->shape().At(1); ++c) {
      for (int64_t out_h = 0; out_h < out_diff_blob->shape().At(2); ++out_h) {
        for (int64_t out_w = 0; out_w < out_diff_blob->shape().At(3); ++out_w) {
          int64_t hstart = out_h * stride_h_ - pad_h_;
          int64_t wstart = out_w * stride_w_ - pad_w_;
          int64_t hend = min(hstart + kernel_h_, height_ + pad_h_);
          int64_t wend = min(wstart + kernel_w_, width_ + pad_w_);
          int64_t pool_size = (hend - hstart) * (wend - wstart);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          hend = std::min(hend, height_);
          wend = std::min(wend, width_);
          const int64_t out_diff_index = out_h * out_width_ + out_w;
          PoolingMethodBackwardFunc_(in_diff_dptr, in_height, in_width, pool_size,
                                     hstart, wstart, hend, wend,
                                     out_diff_dptr, mask_dptr, out_diff_index);
        }
      }
      out_diff_dptr += out_height * out_width;
      mask_dptr += out_height * out_width;
      in_diff_dptr += in_height * in_width;
    }
  }
}

INSTANTIATE_CPU_KERNEL_UTIL_CLASS(PoolingKernelUtil);
INSTANTIATE_KERNEL_CLASS(PoolingKernel);
REGISTER_KERNEL(OperatorConf::kPoolingConf, PoolingKernel);

}  // namespace oneflow
