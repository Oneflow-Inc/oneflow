#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace user_op {

namespace {

template<DeviceType device_type, typename T, typename U>
void CopyTensor(DeviceCtx* ctx, const Tensor* src, Tensor* dst) {
  CHECK_EQ(src->shape(), dst->shape());
  if (device_type == DeviceType::kCPU) {
    CopyElem(src->dptr<T>(), dst->mut_dptr<U>(), src->shape().elem_cnt());
  } else if (device_type == DeviceType::kGPU) {
    CopyElemOnGpu(ctx, src->dptr<T>(), dst->mut_dptr<U>(), src->shape().elem_cnt());
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

#define MAKE_CASE_HANDLER_ENTRY(in_type_pair, out_type_pair)                          \
  {std::make_pair(OF_PP_PAIR_SECOND(in_type_pair), OF_PP_PAIR_SECOND(out_type_pair)), \
   CopyTensor<device_type, OF_PP_PAIR_FIRST(in_type_pair), OF_PP_PAIR_FIRST(out_type_pair)>},

template<DeviceType device_type>
struct CastUtil final {
  static void SwitchCopyTensor(const std::pair<DataType, DataType>& key, DeviceCtx* ctx,
                               const Tensor* src, Tensor* dst) {
    static const std::map<std::pair<DataType, DataType>,
                          std::function<void(DeviceCtx*, const Tensor*, Tensor*)>>
        case_handler{
            // clang-format off
          OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_CASE_HANDLER_ENTRY, POD_DATA_TYPE_SEQ, POD_DATA_TYPE_SEQ)
          MAKE_CASE_HANDLER_ENTRY((float, DataType::kFloat), (float16, DataType::kFloat16))
          MAKE_CASE_HANDLER_ENTRY((float16, DataType::kFloat16), (float, DataType::kFloat))
            // clang-format on
        };
    case_handler.at(key)(ctx, src, dst);
  }
};

template<DeviceType device_type>
class CastKernel final : public OpKernel {
 public:
  CastKernel() = default;
  ~CastKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* output_tenor = ctx->Tensor4ArgNameAndIndex("out", 0);
    CastUtil<device_type>::SwitchCopyTensor(
        std::make_pair(input_tensor->data_type(), output_tenor->data_type()), ctx->device_ctx(),
        input_tensor, output_tenor);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CAST_KERNEL(device)                                                     \
  REGISTER_USER_KERNEL("cast").SetCreateFn<CastKernel<device>>().SetIsMatchedPred(       \
      [](const KernelRegContext& ctx) {                                                  \
        const TensorDesc* output_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0); \
        if (ctx.device_type() == device) { return true; }                                \
        return false;                                                                    \
      });

REGISTER_CAST_KERNEL(DeviceType::kCPU)
REGISTER_CAST_KERNEL(DeviceType::kGPU)

}  // namespace user_op
}  // namespace oneflow
