#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class DynamicConcatKernel final : public user_op::OpKernel {
 public:
  DynamicConcatKernel() = default;
  ~DynamicConcatKernel() = default;

 private:
  void InferShape(user_op::KernelInferContext* ctx) const override {
    const int64_t axis = ctx->Attr<int64_t>("axis");
    DimVector dim_vec;
    for (const auto& in_arg_pair : ctx->inputs()) {
      const ShapeView& input_shape_view =
          ctx->ShapeView4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
      if (dim_vec.size() == 0) {
        input_shape_view.ToDimVector(&dim_vec);
      } else {
        CHECK_EQ(input_shape_view.NumAxes(), dim_vec.size());
        FOR_RANGE(int64_t, i, 0, input_shape_view.NumAxes()) {
          if (i == axis) {
            dim_vec.at(i) += input_shape_view.At(i);
          } else {
            CHECK_EQ(input_shape_view.At(i), dim_vec.at(i));
          }
        }
      }
    }
    ctx->MutShapeView4ArgNameAndIndex("out", 0)->set_shape(Shape(dim_vec));
    // TODO(wenxiao): check concated axis length is less than static length
  }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t axis = ctx->Attr<int64_t>("axis");
    const int64_t out_cols = out_tensor->shape().Count(axis);
    const int64_t rows = out_tensor->shape().elem_cnt() / out_cols;
    CHECK_GT(rows, 0);
    int64_t out_col_offset = 0;
    for (const auto& in_arg_pair : ctx->inputs()) {
      const user_op::Tensor* in_tensor =
          ctx->Tensor4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
      const int64_t in_cols = in_tensor->shape().Count(axis);
      CHECK_EQ(in_tensor->shape().elem_cnt(), rows * in_cols);
      if (in_cols > 0) {
        NewKernelUtil<device_type>::CopyColsRegion(
            ctx->device_ctx(), rows, in_cols, in_tensor->dptr<T>(), 0, in_cols,
            out_tensor->mut_dptr<T>(), out_col_offset, out_cols);
      }
      out_col_offset += in_cols;
    }
    CHECK_EQ(out_col_offset, out_cols);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
bool IsDynamicConcatKernelMatched(const user_op::KernelRegContext& ctx) {
  const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);
  return ctx.device_type() == device_type && out_desc->data_type() == GetDataType<T>::value;
}

}  // namespace

#define REGISTER_DYNAMIC_CONCAT_KERNEL(device, dtype)    \
  REGISTER_USER_KERNEL("dynamic_concat")                 \
      .SetCreateFn<DynamicConcatKernel<device, dtype>>() \
      .SetIsMatchedPred(IsDynamicConcatKernelMatched<device, dtype>);

#define REGISTER_DYNAMIC_CONCAT_KERNEL_WITH_DEVICE_AND_DTYPE_PAIR(device, dtype_pair) \
  REGISTER_DYNAMIC_CONCAT_KERNEL(device, OF_PP_PAIR_FIRST(dtype_pair))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_DYNAMIC_CONCAT_KERNEL_WITH_DEVICE_AND_DTYPE_PAIR,
                                 DEVICE_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ)

REGISTER_DYNAMIC_CONCAT_KERNEL(DeviceType::kGPU, float16)

}  // namespace oneflow
