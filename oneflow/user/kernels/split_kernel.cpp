#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

    namespace {

        template<DeviceType device_type, typename T>
        class SplitKernel final : public user_op::OpKernel {
        public:
            SplitKernel() = default;
            ~SplitKernel() override = default;

        private:
//            void InferShape(user_op::KernelInferContext* ctx) const override {
//                const auto dim = ctx->Attr<int64_t>("axis");
//                const auto sections = ctx->Attr<int64_t>("sections");
//                const ShapeView& in_shape_view = ctx->ShapeView4ArgNameAndIndex("in", 0);
//                const int64_t dim_size = in_shape_view.At(i);
//                const int64_t in_num_axes = in_shape_view.NumAxes();
////                int64_t total_dim_size = 0;
////                const int64_t like_num_axes = ctx->ShapeView4ArgNameAndIndex("like", 0).NumAxes();
//                CHECK_GT(dim, 0);
//
////                CHECK_LE(like_num_axes, in_num_axes);
//                CHECK_LT(dim, in_num_axes);
//                FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
//                    const ShapeView& like_shape_view = ctx->ShapeView4ArgNameAndIndex("like", i);
//                    CHECK_EQ(like_shape_view.NumAxes(), like_num_axes);
//                    FOR_RANGE(int64_t, j, 0, like_num_axes) {
//                        if (j == axis) {
//                            total_dim_size += like_shape_view.At(j);
//                        } else {
//                            CHECK_EQ(like_shape_view.At(j), in_shape_view.At(j));
//                        }
//                    }
//                    if (ctx->TensorDesc4ArgNameAndIndex("out", i)->is_dynamic()) {
//                        auto* mut_shape_view = ctx->MutShapeView4ArgNameAndIndex("out", i);
//                        CHECK_NOTNULL(mut_shape_view);
//                        DimVector out_i_dim_vec;
//                        like_shape_view.ToDimVector(&out_i_dim_vec);
//                        FOR_RANGE(int64_t, j, like_num_axes, in_num_axes) {
//                            out_i_dim_vec.push_back(in_shape_view.At(j));
//                        }
//                        mut_shape_view->set_shape(Shape(out_i_dim_vec));
//                    }
//                }
//                CHECK_EQ(total_dim_size, in_shape_view.At(axis));
//            }

            void Compute(user_op::KernelComputeContext* ctx) const override {
                const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
                const auto dim = ctx->Attr<int64_t>("axis");
                CHECK_GE(dim, 0);
                const auto sections = ctx->Attr<int64_t>("sections");
                CHECK_GE(sections, 0);
                const auto sizes = ctx->Attr<std::vector<int64_t>>("sizes");
                const ShapeView& in_shape_view = ctx->ShapeView4ArgNameAndIndex("in", 0);
                const int64_t dim_size = in_shape_view.At(i);
                if(sections != nullptr)
                {
                    const int64_t min_split_size = dim_size / sections;
                    const int64_t num_splits_one_extra = dim_size % sections;
                    int64_t start_idx = 0;
                    FOR_RANGE(int64_t, i, 0, sections){
                        int64_t split_size = (split_idx < num_splits_one_extra) ? (min_split_size + 1) : min_split_size;
                        splits[split_idx] = at::slice(self, dim_, start_idx, start_idx + split_size);
                        start_idx += split_size;
                    }
                }

                const int64_t in_cols = in_tensor->shape().Count(axis);
                const int64_t rows = in_tensor->shape().elem_cnt() / in_cols;
                CHECK_GT(rows, 0);
                int64_t in_col_offset = 0;
                for (const auto& out_arg_pair : ctx->outputs()) {
                    user_op::Tensor* out_tensor =
                            ctx->Tensor4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second);
                    const int64_t out_cols = out_tensor->shape().Count(axis);
                    CHECK_EQ(out_tensor->shape().elem_cnt(), rows * out_cols);
                    if (out_cols > 0) {
                        NewKernelUtil<device_type>::CopyColsRegion(ctx->device_ctx(), rows, out_cols,
                                                                   in_tensor->dptr<T>(), in_col_offset, in_cols,
                                                                   out_tensor->mut_dptr<T>(), 0, out_cols);
                    }
                    in_col_offset += out_cols;
                }
                CHECK_EQ(in_col_offset, in_cols);
            }

            bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
        };

    }  // namespace

#define REGISTER_SPLIT_LIKE_KERNEL(device, dtype)          \
  REGISTER_USER_KERNEL("split_like")                       \
      .SetCreateFn<SplitLikeKernel<device, dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

#define REGISTER_SPLIT_LIKE_KERNEL_WITH_DEVICE(device) \
  REGISTER_SPLIT_LIKE_KERNEL(device, float)            \
  REGISTER_SPLIT_LIKE_KERNEL(device, double)           \
  REGISTER_SPLIT_LIKE_KERNEL(device, int8_t)           \
  REGISTER_SPLIT_LIKE_KERNEL(device, int32_t)          \
  REGISTER_SPLIT_LIKE_KERNEL(device, int64_t)

    REGISTER_SPLIT_LIKE_KERNEL_WITH_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
    REGISTER_SPLIT_LIKE_KERNEL_WITH_DEVICE(DeviceType::kGPU)
REGISTER_SPLIT_LIKE_KERNEL(DeviceType::kGPU, float16)
#endif

}  // namespace oneflow
