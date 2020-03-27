#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ReduceAnyKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceAnyKernel);
  ReduceAnyKernel() = default;
  ~ReduceAnyKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    Blob* fw_tmp_blob = BnInOp2Blob("fw_tmp");
    const ReduceAnyOpConf& conf = this->op_conf().reduce_any_conf();
    const Shape& reduced_shape =
        conf.axis().empty()
            ? Shape::Ones(in_blob->shape().NumAxes())
            : CreateReducedShape(in_blob->shape(), {conf.axis().begin(), conf.axis().end()});
    NdarrayUtil<device_type, T>::ReduceAny(
        ctx.device_ctx, XpuVarNdarray<T>(reduced_shape, out_blob->mut_dptr<T>()),
        XpuVarNdarray<const T>(in_blob, in_blob->shape().NumAxes()),
        XpuVarNdarray<T>(fw_tmp_blob, in_blob->shape().NumAxes()));
  }
};

ADD_DEFAULT_KERNEL_CREATOR_WITH_GPU_HALF(OperatorConf::kReduceAnyConf, ReduceAnyKernel,
                                         ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow