#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/categorical_ordinal_encode_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class CategoricalOrdinalEncodeKernel final : public user_op::OpKernel {
 public:
  explicit CategoricalOrdinalEncodeKernel(user_op::KernelInitContext* ctx)
      : user_op::OpKernel(ctx) {}
  ~CategoricalOrdinalEncodeKernel() override = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    bool hash_precomputed = ctx->GetAttr<bool>("hash_precomputed");
    CHECK(hash_precomputed);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* table = ctx->Tensor4ArgNameAndIndex("table", 0);
    user_op::Tensor* size = ctx->Tensor4ArgNameAndIndex("size", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t table_elem_cnt = table->shape().elem_cnt();
    CHECK_EQ(table_elem_cnt % 2, 0);
    const int64_t capacity = table_elem_cnt / 2;
    CategoricalOrdinalEncodeKernelUtil<device_type, T>::Encode(
        ctx->device_ctx(), capacity, table->mut_dptr<T>(), size->mut_dptr<T>(),
        in->shape().elem_cnt(), in->dptr<T>(), out->mut_dptr<T>());
  }
};

#define REGISTER_CATEGORICAL_ORDINAL_ENCODE_KERNEL(device, proto_type, cpp_type) \
  REGISTER_USER_KERNEL("CategoricalOrdinalEncode")                               \
      .SetCreateFn([](user_op::KernelInitContext* ctx) {                         \
        return new CategoricalOrdinalEncodeKernel<device, cpp_type>(ctx);        \
      })                                                                         \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {               \
        const user_op::TensorDesc* in = ctx.TensorDesc4ArgNameAndIndex("in", 0); \
        return ctx.device_type() == device && in->data_type() == proto_type;     \
      });

REGISTER_CATEGORICAL_ORDINAL_ENCODE_KERNEL(DeviceType::kCPU, DataType::kInt32, int32_t);
REGISTER_CATEGORICAL_ORDINAL_ENCODE_KERNEL(DeviceType::kCPU, DataType::kInt64, int64_t);
REGISTER_CATEGORICAL_ORDINAL_ENCODE_KERNEL(DeviceType::kGPU, DataType::kInt32, int32_t);
REGISTER_CATEGORICAL_ORDINAL_ENCODE_KERNEL(DeviceType::kGPU, DataType::kInt64, int64_t);

}  // namespace oneflow
