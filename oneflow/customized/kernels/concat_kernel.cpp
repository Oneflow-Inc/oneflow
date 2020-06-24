#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {
namespace {

template<DeviceType device_type, typename T>
class ConcatKernel final : public user_op::OpKernel {
 public:
  ConcatKernel() = default;
  ~ConcatKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t axis = ctx->Attr<int32_t>("axis");
    const int64_t row_num = out->shape().elem_cnt() / out->shape().Count(axis);
    const int64_t out_col_num = out->shape().Count(axis);
    int64_t out_col_offset = 0;
    for (int32_t i = 0; i < ctx->inputs().size(); ++i) {
      const user_op::Tensor* in_i = ctx->Tensor4ArgNameAndIndex("in", i);
      const int64_t in_col_num = in_i->shape().Count(axis);
      CHECK_EQ(in_i->shape().elem_cnt(), row_num * in_col_num);
      CHECK_EQ(in_i->data_type(), out->data_type());
      if (row_num * in_col_num > 0) {
        NewKernelUtil<device_type>::CopyColsRegion(ctx->device_ctx(), row_num, in_col_num,
                                                   in_i->dptr<T>(), 0, in_col_num,
                                                   out->mut_dptr<T>(), out_col_offset, out_col_num);
        out_col_offset += in_col_num;
      }
    }
    CHECK_EQ(out_col_offset, out_col_num);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_CONCAT_KERNEL(device_type_v, dtype_pair)                       \
  REGISTER_USER_KERNEL("concat")                                                \
      .SetCreateFn<ConcatKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>>() \
      .SetIsMatchedHob(user_op::HobDeviceType() == device_type_v                \
                       & user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CONCAT_KERNEL, DEVICE_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ)

REGISTER_USER_KERNEL("concat")
    .SetCreateFn<ConcatKernel<DeviceType::kGPU, float16>>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kGPU
                     & user_op::HobDataType("out", 0) == DataType::kFloat16);
}  // namespace oneflow
