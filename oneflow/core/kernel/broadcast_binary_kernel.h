#ifndef ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

Shape CalcBroadcastShape(const Shape& a_shape, const Shape& b_shape) {
  const int32_t num_axis = std::max(a_shape.NumAxes(), b_shape.NumAxes());
  const Shape& extended_a_shape = a_shape.CreateLeftExtendedShape(num_axis);
  const Shape& extended_b_shape = b_shape.CreateLeftExtendedShape(num_axis);
  Shape broadcast_shape(extended_a_shape);
  FOR_RANGE(int32_t, i, 0, num_axis) {
    CHECK(extended_a_shape.At(i) == 1 || extended_b_shape.At(i) == 1
          || extended_a_shape.At(i) == extended_b_shape.At(i));
    broadcast_shape.Set(i, std::max(a_shape.At(i), b_shape.At(i)));
  }
  return broadcast_shape;
}

}  // namespace

template<DeviceType device_type, typename T>
class BroadcastBinaryKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastBinaryKernel);
  BroadcastBinaryKernel() = default;
  ~BroadcastBinaryKernel() = default;

 private:
  virtual void ForwardDataContent(const KernelCtx&,
                                  std::function<Blob*(const std::string&)>) const {
    UNIMPLEMENTED();
  }
  virtual void BackwardDataContent(const KernelCtx&,
                                   std::function<Blob*(const std::string&)>) const {
    UNIMPLEMENTED();
  }
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    BnInOp2Blob("out")->set_dim0_valid_num(
        0, CalcBroadcastShape(BnInOp2Blob("a")->shape(), BnInOp2Blob("b")->shape()).At(0));
  }
  void ForwardInstanceShape(const KernelCtx&,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    std::vector<int64_t> dim_vec =
        CalcBroadcastShape(BnInOp2Blob("a")->shape(), BnInOp2Blob("b")->shape()).dim_vec();
    dim_vec.erase(dim_vec.begin());
    BnInOp2Blob("out")->set_instance_shape(Shape(dim_vec));
  }
  void BackwardInDiffDim0ValidNum(const KernelCtx& ctx,
                                  std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    const Blob* a_blob = BnInOp2Blob("a");
    const Blob* b_blob = BnInOp2Blob("b");
    Blob* a_diff_blob = BnInOp2Blob(GenDiffBn("a"));
    Blob* b_diff_blob = BnInOp2Blob(GenDiffBn("b"));
    if (a_diff_blob && a_diff_blob->has_dim0_valid_num_field()) {
      CHECK(a_blob->has_dim0_valid_num_field());
      a_diff_blob->CopyDim0ValidNumFrom(ctx.device_ctx, a_blob);
    }
    if (b_diff_blob && b_diff_blob->has_dim0_valid_num_field()) {
      CHECK(b_blob->has_dim0_valid_num_field());
      b_diff_blob->CopyDim0ValidNumFrom(ctx.device_ctx, b_blob);
    }
  }
  void BackwardInstanceShape(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    const Blob* a_blob = BnInOp2Blob("a");
    const Blob* b_blob = BnInOp2Blob("b");
    Blob* a_diff_blob = BnInOp2Blob(GenDiffBn("a"));
    Blob* b_diff_blob = BnInOp2Blob(GenDiffBn("b"));
    if (a_diff_blob && a_diff_blob->has_instance_shape_field()) {
      CHECK(a_blob->has_instance_shape_field());
      a_diff_blob->CopyInstanceShapeFrom(ctx.device_ctx, a_blob);
    }
    if (b_diff_blob && b_diff_blob->has_instance_shape_field()) {
      CHECK(b_blob->has_instance_shape_field());
      b_diff_blob->CopyInstanceShapeFrom(ctx.device_ctx, b_blob);
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_KERNEL_H_
