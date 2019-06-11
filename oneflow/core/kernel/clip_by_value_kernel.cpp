#include "oneflow/core/kernel/clip_by_value_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ClipByValueKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* clip_mask_blob = BnInOp2Blob("clip_mask");
  Blob* out_blob = BnInOp2Blob("out");

  const Shape shape = in_blob->shape();
  CHECK_EQ(clip_mask_blob->shape(), shape);
  CHECK_EQ(out_blob->shape(), shape);
  const ClipByValueOpConf& conf = this->op_conf().clip_by_value_conf();
  ClipByValueUtil<device_type, T>::Forward(
      ctx.device_ctx, shape.elem_cnt(), in_blob->dptr<T>(), static_cast<T>(conf.min_val()),
      static_cast<T>(conf.max_val()), clip_mask_blob->mut_dptr<int8_t>(), out_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void ClipByValueKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  const Blob* clip_mask_blob = BnInOp2Blob("clip_mask");
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));

  const Shape shape = out_diff_blob->shape();
  CHECK_EQ(clip_mask_blob->shape(), shape);
  CHECK_EQ(in_diff_blob->shape(), shape);
  ClipByValueUtil<device_type, T>::Backward(
      ctx.device_ctx, shape.elem_cnt(), out_diff_blob->dptr<T>(), clip_mask_blob->dptr<int8_t>(),
      in_diff_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void ClipByValueKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx&, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* clip_mask = BnInOp2Blob("clip_mask");
  Blob* out = BnInOp2Blob("out");
  CHECK(in_blob->has_dim0_valid_num_field());
  CHECK(clip_mask->has_dim0_valid_num_field());
  CHECK(out->has_dim0_valid_num_field());
  clip_mask->set_dim0_valid_num(0, in_blob->dim0_valid_num(0));
  out->set_dim0_valid_num(0, in_blob->dim0_valid_num(0));
}

template<DeviceType device_type, typename T>
void ClipByValueKernel<device_type, T>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* clip_mask = BnInOp2Blob("clip_mask");
  Blob* out = BnInOp2Blob("out");
  CHECK(in_blob->has_instance_shape_field());
  CHECK(clip_mask->has_instance_shape_field());
  CHECK(out->has_instance_shape_field());
  clip_mask->set_instance_shape(in_blob->instance_shape());
  out->set_instance_shape(in_blob->instance_shape());
}

template<typename T>
struct ClipByValueUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const T* in_ptr, const T min_val,
                      const T max_val, int8_t* clip_mask_ptr, T* out_ptr) {
    UNIMPLEMENTED();
  }
  static void Backward(DeviceCtx* ctx, const int64_t elem_cnt, const T* out_diff_ptr,
                       const int8_t* clip_mask_ptr, T* in_diff_ptr) {
    UNIMPLEMENTED();
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kClipByValueConf, ClipByValueKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
