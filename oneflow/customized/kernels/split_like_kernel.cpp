#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {
namespace {

template<DeviceType device_type, typename T>
class SplitLikeKernel final : public user_op::OpKernel {
 public:
  SplitLikeKernel() = default;
  ~SplitLikeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const int32_t axis = ctx->Attr<int32_t>("axis");
    const int64_t row_num = in->shape().elem_cnt() / in->shape().Count(axis);
    const int64_t in_col_num = in->shape().Count(axis);
    int64_t in_col_offset = 0;
    for (int32_t i = 0; i < ctx->outputs().size(); ++i) {
      user_op::Tensor* out_i = ctx->Tensor4ArgNameAndIndex("out", i);
      const int64_t out_col_num = out_i->shape().Count(axis);
      CHECK_EQ(out_i->shape().elem_cnt(), row_num * out_col_num);
      CHECK_EQ(out_i->data_type(), in->data_type());
      if (row_num * out_col_num > 0) {
        NewKernelUtil<device_type>::CopyColsRegion(ctx->device_ctx(), row_num, out_col_num,
                                                   in->dptr<T>(), in_col_offset, in_col_num,
                                                   out_i->mut_dptr<T>(), 0, out_col_num);
      }
      in_col_offset += out_col_num;
    }
    CHECK_EQ(in_col_offset, in_col_num);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
}  // namespace

#define REGISTER_SPLIT_LIKE_KERNEL(device_type_v, dtype_pair)                      \
  REGISTER_USER_KERNEL("split_like")                                               \
      .SetCreateFn<SplitLikeKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>>() \
      .SetIsMatchedHob(user_op::HobDeviceType() == device_type_v                   \
                       & user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPLIT_LIKE_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

REGISTER_USER_KERNEL("split_like")
    .SetCreateFn<SplitLikeKernel<DeviceType::kGPU, float16>>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kGPU
                     & user_op::HobDataType("out", 0) == DataType::kFloat16);

}  // namespace oneflow
