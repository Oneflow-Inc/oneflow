#ifndef ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

Shape CalcOutShape(const Shape& a_shape, const Shape& b_shape) {
  const int32_t out_num_axis = std::max(a_shape.NumAxes(), b_shape.NumAxes());
  const Shape& extended_a_shape = a_shape.CreateLeftExtendedShape(out_num_axis);
  const Shape& extended_b_shape = b_shape.CreateLeftExtendedShape(out_num_axis);
  Shape out_shape(extended_a_shape);
  FOR_RANGE(int32_t, i, 0, out_num_axis) {
    CHECK(extended_a_shape.At(i) == 1 || extended_b_shape.At(i) == 1
          || extended_a_shape.At(i) == extended_b_shape.At(i));
    out_shape.Set(i, std::max(a_shape.At(i), b_shape.At(i)));
  }
  return out_shape;
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
        0, CalcOutShape(BnInOp2Blob("a")->shape(), BnInOp2Blob("b")->shape()).At(0));
  }
  void ForwardInstanceShape(const KernelCtx&,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    std::vector<int64_t> dim_vec =
        CalcOutShape(BnInOp2Blob("a")->shape(), BnInOp2Blob("b")->shape()).dim_vec();
    dim_vec.erase(dim_vec.begin());
    BnInOp2Blob("out")->set_instance_shape(Shape(dim_vec));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_KERNEL_H_
