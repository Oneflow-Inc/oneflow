#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/slice_util.h"

namespace oneflow {

    namespace {
        constexpr size_t
        kSliceMaxDims = 8;

        SliceParams ConstructSplitParams(const user_op::Tensor *entire, const user_op::Tensor *sliced,
                                         const int64_t dim, int64_t start_idx, int64_t end_idx) {
//            const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
//            const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
//            const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");

            const int64_t ndim = entire->shape().NumAxes();
            CHECK_LE(ndim, kSliceMaxDims);
            CHECK_EQ(sliced->shape().NumAxes(), ndim);
//            CHECK_EQ(start_vec.size(), ndim);
//            CHECK_EQ(stop_vec.size(), ndim);
//            CHECK_EQ(step_vec.size(), ndim);

            SliceParams params;
            std::memset(&params, 0, sizeof(SliceParams));
            params.ndim = ndim;
            FOR_RANGE(int, i, 0, params.ndim)
            {
                const int64_t dim_size = entire->shape().At(i);
                const int64_t slice_size = sliced->shape().At(i);
//                const int64_t step = step_vec.at(i);
//                CHECK_NE(step, 0);
                CHECK_GE(start_idx, 0);
                CHECK_LT(start_idx, dim_size);
                CHECK_GE(end_idx, 0);
                CHECK_LT(end_idx, dim_size);
                const int64_t dim_start_idx = i == dim ? start_idx : 0;
                const int64_t dim_end_idx = i == dim ? end_idx : dim_size - 1;
                const int64_t start = RegulateSliceStart(dim_start_idx, dim_size);
                const int64_t stop = RegulateSliceStop(dim_end_idx, dim_size);

//                const int64_t start = RegulateSliceStart(start_vec.at(i), dim_size);
//                const int64_t stop = RegulateSliceStop(stop_vec.at(i), dim_size);
//                if (step > 0) {
                CHECK_LT(start + slice_size - 1, stop);
//                } else {
//                    CHECK_GT(start + step * (slice_size - 1), stop);
//                }
                params.dims[i] = dim_size;
                params.start[i] = start;
                params.step[i] = 1;
                params.size[i] = slice_size;
            }
            return params;
        }


//        template<typename T>
//        template<DeviceType device_type, typename T>
//        void SplitSectionUtil(user_op::KernelComputeContext *ctx) {
//
//        }

        template<DeviceType device_type, typename T>
        class SplitKernel final : public user_op::OpKernel {
        public:
            SplitKernel() = default;

            ~SplitKernel() override = default;

        private:
//            void InferShape(user_op::KernelInferContext *ctx) const override {
//                const auto dim = ctx->Attr<int64_t>("axis");
//                const auto sections = ctx->Attr<int64_t>("sections");
//                const ShapeView &in_shape_view = ctx->ShapeView4ArgNameAndIndex("in", 0);
//                int64_t total_dim_size = 0;
////        const int64_t like_num_axes = ctx->ShapeView4ArgNameAndIndex("like", 0).NumAxes();
//                const int64_t in_num_axes = in_shape_view.NumAxes();
////        CHECK_LE(like_num_axes, in_num_axes);
////        CHECK_LT(axis, like_num_axes);
//                FOR_RANGE(int32_t, i, 0, ctx->outputs().size())
//                    {
//                        ////            const ShapeView& like_shape_view = ctx->ShapeView4ArgNameAndIndex("like", i);
//                        ////            CHECK_EQ(like_shape_view.NumAxes(), like_num_axes);
//                        //            FOR_RANGE(int64_t, j, 0, like_num_axes) {
//                        //                if (j == axis) {
//                        //                total_dim_size += like_shape_view.At(j);
//                        //                    } else {
//                        //                    CHECK_EQ(like_shape_view.At(j), in_shape_view.At(j));
//                        //                }
//                        //            }
//                        if (ctx->TensorDesc4ArgNameAndIndex("out", i)->is_dynamic()) {
//                            auto *mut_shape_view = ctx->MutShapeView4ArgNameAndIndex("out", i);
//                            CHECK_NOTNULL(mut_shape_view);
//                            DimVector out_i_dim_vec;
//                            in_shape_view.ToDimVector(&out_i_dim_vec);
//                            FOR_RANGE(int64_t, j, 0, in_num_axes)
//                            {
//                                if (dim == j)
//                                    out_i_dim_vec.push_back(sections);
//                                else
//                                    out_i_dim_vec.push_back(in_shape_view.At(j));
//                            }
//                            mut_shape_view->set_shape(Shape(out_i_dim_vec));
//                        }
//                    }
//
//                CHECK_EQ(total_dim_size, in_shape_view.At(dim));
//            }

            void Compute(user_op::KernelComputeContext *ctx) const override {
//                const auto sizes = ctx->Attr < std::vector < int64_t >> ("sizes");
//                const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
//                const auto dim = ctx->Attr<int64_t>("axis");
//                CHECK_GE(dim, 0);
//                const auto sections = ctx->Attr<int64_t>("sections");
//                const auto sizes = ctx->Attr<std::vector<int64_t>>("sizes");
//                const ShapeView& in_shape_view = ctx->ShapeView4ArgNameAndIndex("in", 0);
//                const int64_t dim_size = in_shape_view.At(i);
//                SplitSectionUtil(ctx);
//                    CHECK_GE(sections, 0);


//                const int64_t in_cols = in_tensor->shape().Count(axis);
//                const int64_t rows = in_tensor->shape().elem_cnt() / in_cols;
//                CHECK_GT(rows, 0);
//                int64_t in_col_offset = 0;
//                for (const auto& out_arg_pair : ctx->outputs()) {
//                    user_op::Tensor* out_tensor =
//                            ctx->Tensor4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second);
//                    const int64_t out_cols = out_tensor->shape().Count(axis);
//                    CHECK_EQ(out_tensor->shape().elem_cnt(), rows * out_cols);
//                    if (out_cols > 0) {
//                        NewKernelUtil<device_type>::CopyColsRegion(ctx->device_ctx(), rows, out_cols,
//                                                                   in_tensor->dptr<T>(), in_col_offset, in_cols,
//                                                                   out_tensor->mut_dptr<T>(), 0, out_cols);
//                    }
//                    in_col_offset += out_cols;
//                }
//                CHECK_EQ(in_col_offset, in_cols);

                const user_op::Tensor *in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
                const auto dim = ctx->Attr<int64_t>("axis");
                CHECK_GE(dim, 0);
                const auto sections = ctx->Attr<int64_t>("sections");
                CHECK_GE(sections, 0);

                const int64_t dim_size = in_tensor->shape().Count(dim);
                const int64_t min_split_size = dim_size / sections;
                const int64_t num_splits_one_extra = dim_size % sections;
                const int64_t num_splits = min_split_size + (num_splits_one_extra > 0 ? 1 : 0);
                int64_t start_idx = 0;
                FOR_RANGE(int64_t, split_idx, 0, num_splits)
                {
                    user_op::Tensor *out_i = ctx->Tensor4ArgNameAndIndex("out", split_idx);
                    const int64_t end_idx = split_idx >= min_split_size? start_idx + num_splits_one_extra - 1 : start_idx + sections - 1;
                    SliceParams params = ConstructSplitParams(in_tensor, out_i, dim, start_idx, end_idx);
                    SliceKernelUtil<device_type, T>::Forward(ctx->device_ctx(), params, in_tensor->dptr<T>(),
                                                             out_i->mut_dptr<T>());
//                int64_t split_size = (split_idx < num_splits_one_extra) ? (min_split_size + 1) : min_split_size;
//                splits[split_idx] = at::slice(self, dim_, start_idx, start_idx + split_size);
                    start_idx += sections;
                }
            }

            bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
        };

    }  // namespace

#define REGISTER_SPLIT_KERNEL(device, dtype)          \
  REGISTER_USER_KERNEL("split")                       \
      .SetCreateFn<SplitKernel<device, dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

#define REGISTER_SPLIT_KERNEL_WITH_DEVICE(device) \
  REGISTER_SPLIT_KERNEL(device, float)            \
  REGISTER_SPLIT_KERNEL(device, double)           \
  REGISTER_SPLIT_KERNEL(device, int8_t)           \
  REGISTER_SPLIT_KERNEL(device, int32_t)          \
  REGISTER_SPLIT_KERNEL(device, int64_t)

    REGISTER_SPLIT_KERNEL_WITH_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
    REGISTER_SPLIT_KERNEL_WITH_DEVICE(DeviceType::kGPU)
REGISTER_SPLIT_KERNEL(DeviceType::kGPU, float16)
#endif

}  // namespace oneflow
