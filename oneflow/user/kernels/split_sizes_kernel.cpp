#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/slice_util.h"

namespace oneflow {

    namespace {
        constexpr size_t kSliceMaxDims = 8;

        SliceParams ConstructSplitSizesParams(const user_op::Tensor *entire, const user_op::Tensor *sliced,
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
    }  // namespace


        template<DeviceType device_type, typename T>
        class SplitSizesKernel final : public user_op::OpKernel {
        public:
            SplitSizesKernel() = default;

            ~SplitSizesKernel() override = default;

        private:

            void Compute(user_op::KernelComputeContext *ctx) const override {
//
                const user_op::Tensor *in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
                const auto dim = ctx->Attr<int64_t>("axis");
                CHECK_GE(dim, 0);
                const auto sizes_list = ctx->Attr<std::vector<int64_t>>("sizes");
                int64_t start_idx = 0;
                FOR_RANGE(int64_t, split_idx, 0, sizes_list.size())
                {
                    user_op::Tensor *out_i = ctx->Tensor4ArgNameAndIndex("out", split_idx);
                    SliceParams params = ConstructSplitSizesParams(in_tensor, out_i, dim, start_idx, start_idx + sizes_list[split_idx] - 1);
                    SliceKernelUtil<device_type, T>::Forward(ctx->device_ctx(), params, in_tensor->dptr<T>(),
                                                             out_i->mut_dptr<T>());
                    start_idx += sizes_list[split_idx];
                }
            }

            bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
        };



#define REGISTER_SPLIT_SIZES_KERNEL(device, dtype)          \
  REGISTER_USER_KERNEL("split_sizes")                       \
      .SetCreateFn<SplitSizesKernel<device, dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

#define REGISTER_SPLIT_SIZES_KERNEL_WITH_DEVICE(device) \
  REGISTER_SPLIT_SIZES_KERNEL(device, float)            \
  REGISTER_SPLIT_SIZES_KERNEL(device, double)           \
  REGISTER_SPLIT_SIZES_KERNEL(device, int8_t)           \
  REGISTER_SPLIT_SIZES_KERNEL(device, int32_t)          \
  REGISTER_SPLIT_SIZES_KERNEL(device, int64_t)

    REGISTER_SPLIT_SIZES_KERNEL_WITH_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
    REGISTER_SPLIT_SIZES_KERNEL_WITH_DEVICE(DeviceType::kGPU)
REGISTER_SPLIT_SIZES_KERNEL(DeviceType::kGPU, float16)
#endif

}  // namespace oneflow
