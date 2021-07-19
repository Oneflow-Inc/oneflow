#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/slice_util.h"

namespace oneflow {

    namespace {
        constexpr size_t
        kSliceMaxDims = 8;

        SliceParams ConstructSplitParams(const user_op::Tensor *entire, const user_op::Tensor *sliced,
                                         const int64_t dim, int64_t start_idx, int64_t end_idx) {

            const int64_t ndim = entire->shape().NumAxes();
            CHECK_LE(ndim, kSliceMaxDims);
            CHECK_EQ(sliced->shape().NumAxes(), ndim);

            SliceParams params;
            std::memset(&params, 0, sizeof(SliceParams));
            params.ndim = ndim;
            FOR_RANGE(int, i, 0, params.ndim)
            {
                const int64_t dim_size = entire->shape().At(i);
                const int64_t slice_size = sliced->shape().At(i);
                CHECK_GE(start_idx, 0);
                CHECK_LT(start_idx, dim_size);
                CHECK_GE(end_idx, 0);
                CHECK_LT(end_idx, dim_size);
                const int64_t dim_start_idx = i == dim ? start_idx : 0;
                const int64_t dim_end_idx = i == dim ? end_idx : dim_size - 1;
                const int64_t start = RegulateSliceStart(dim_start_idx, dim_size);
                const int64_t stop = RegulateSliceStop(dim_end_idx, dim_size);

                CHECK_LT(start + slice_size - 1, stop);

                params.dims[i] = dim_size;
                params.start[i] = start;
                params.step[i] = 1;
                params.size[i] = slice_size;
            }
            return params;
        }



        template<DeviceType device_type, typename T>
        class SplitKernel final : public user_op::OpKernel {
        public:
            SplitKernel() = default;

            ~SplitKernel() override = default;

        private:

            void Compute(user_op::KernelComputeContext *ctx) const override {
//
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
