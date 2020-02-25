#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/customized/ops/slice_util.h"

namespace oneflow {

namespace {

constexpr size_t kSliceMaxDims = 8;

struct SliceGpuParams {
  int64_t ndims;
  int64_t dims[kSliceMaxDims];
  int64_t sliced_dims[kSliceMaxDims];
  int64_t begin[kSliceMaxDims];
  int64_t end[kSliceMaxDims];
  int64_t stride[kSliceMaxDims];
};

SliceGpuParams ConstructSliceGpuParams(user_op::KernelContext* ctx, const user_op::Tensor* entire,
                                       const user_op::Tensor* sliced) {
  auto begin_vec = ctx->GetAttr<std::vector<int64_t>>("begin");
  auto end_vec = ctx->GetAttr<std::vector<int64_t>>("end");
  auto stride_vec = ctx->GetAttr<std::vector<int64_t>>("stride");
  CHECK_LE(entire->shape().NumAxes(), kSliceMaxDims);
  CHECK_EQ(entire->shape().NumAxes(), sliced->shape().NumAxes());
  CHECK_EQ(entire->shape().NumAxes(), begin_vec.size() + 1);
  CHECK_EQ(entire->shape().NumAxes(), end_vec.size() + 1);
  CHECK_EQ(entire->shape().NumAxes(), stride_vec.size() + 1);

  SliceGpuParams params;
  std::memset(&params, 0, sizeof(SliceGpuParams));
  params.ndims = entire->shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, params.ndims) {
    params.dims[i] = entire->shape().At(i);
    params.sliced_dims[i] = sliced->shape().At(i);
    if (i == 0) {
      params.begin[i] = 0;
      params.end[i] = params.dims[i];
      params.stride[i] = 1;
    } else {
      params.begin[i] = RegulateSliceIndex(begin_vec.at(i - 1), params.dims[i]);
      params.end[i] = RegulateSliceIndex(end_vec.at(i - 1), params.dims[i]);
      params.stride[i] = stride_vec.at(i - 1);
      CHECK_NE(params.stride[i], 0);
      if (params.stride[i] > 0) {
        CHECK_LT(params.begin[i], params.end[i]);
      } else {
        CHECK_GT(params.begin[i], params.end[i]);
      }
    }
  }
  return params;
}

__device__ __forceinline__ void OffsetToDimCoords(const int64_t offset, const int64_t ndims,
                                                  const int64_t* dims, int64_t* dim_coords) {
  int64_t divisor = offset;
#pragma unroll
  for (int64_t i = ndims - 1; i >= 0; --i) {
    dim_coords[i] = divisor % dims[i];
    divisor /= dims[i];
  }
}

__device__ __forceinline__ int64_t DimCoordsToOffset(const int64_t ndims, const int64_t* dims,
                                                     const int64_t* dim_coords) {
  int64_t offset = 0;
  int64_t product = 1;
#pragma unroll
  for (int64_t i = ndims - 1; i >= 0; --i) {
    offset += dim_coords[i] * product;
    product *= dims[i];
  }
  return offset;
}

template<typename T>
__global__ void SliceForwardGpu(const int n, SliceGpuParams params, const T* entire, T* part) {
  int64_t dim_coords[kSliceMaxDims];
  CUDA_1D_KERNEL_LOOP(i, n) {
    OffsetToDimCoords(i, params.ndims, params.sliced_dims, dim_coords);
#pragma unroll
    for (int64_t i = 0; i < params.ndims; ++i) {
      dim_coords[i] = params.begin[i] + params.stride[i] * dim_coords[i];
      assert(dim_coords[i] < params.end[i]);
    }
    int64_t offset = DimCoordsToOffset(params.ndims, params.dims, dim_coords);
    part[i] = entire[offset];
  }
}

template<typename T>
__global__ void SliceBackwardGpu(const int n, SliceGpuParams params, const T* part, T* entire) {
  int64_t dim_coords[kSliceMaxDims];
  CUDA_1D_KERNEL_LOOP(i, n) {
    OffsetToDimCoords(i, params.ndims, params.sliced_dims, dim_coords);
#pragma unroll
    for (int64_t i = 0; i < params.ndims; ++i) {
      dim_coords[i] = params.begin[i] + params.stride[i] * dim_coords[i];
      assert(dim_coords[i] < params.end[i]);
    }
    int64_t offset = DimCoordsToOffset(params.ndims, params.dims, dim_coords);
    entire[offset] = part[i];
  }
}

}  // namespace

template<typename T>
class SliceGpuKernel final : public user_op::OpKernel {
 public:
  SliceGpuKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  SliceGpuKernel() = default;
  ~SliceGpuKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto params = ConstructSliceGpuParams(ctx, input, output);
    int64_t elem_cnt = output->shape().elem_cnt();
    SliceForwardGpu<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                         ctx->device_ctx()->cuda_stream()>>>(elem_cnt, params, input->dptr<T>(),
                                                             output->mut_dptr<T>());
  }
};

template<typename T>
class SliceGradGpuKernel final : public user_op::OpKernel {
 public:
  SliceGradGpuKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  SliceGradGpuKernel() = default;
  ~SliceGradGpuKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    size_t dx_byte_size = dx->shape().elem_cnt() * sizeof(T);
    Memset<DeviceType::kGPU>(ctx->device_ctx(), dx->mut_dptr<T>(), 0, dx_byte_size);
    auto params = ConstructSliceGpuParams(ctx, dx, dy);
    int64_t elem_cnt = dy->shape().elem_cnt();
    SliceBackwardGpu<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
           ctx->device_ctx()->cuda_stream()>>>(elem_cnt, params, dy->dptr<T>(), dx->mut_dptr<T>());
  }
};

#define REGISTER_SLICE_GPU_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("slice_v2")                                                    \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {               \
        return new SliceGpuKernel<dtype>(ctx);                                        \
      })                                                                              \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {           \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);   \
        if (ctx.device() == DeviceType::kGPU                                          \
            && y_desc->data_type() == GetDataType<dtype>::value) {                    \
          return true;                                                                \
        }                                                                             \
        return false;                                                                 \
      });                                                                             \
  REGISTER_USER_KERNEL("slice_grad_v2")                                               \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {               \
        return new SliceGradGpuKernel<dtype>(ctx);                                    \
      })                                                                              \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {           \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        if (ctx.device() == DeviceType::kGPU                                          \
            && dx_desc->data_type() == GetDataType<dtype>::value) {                   \
          return true;                                                                \
        }                                                                             \
        return false;                                                                 \
      });

REGISTER_SLICE_GPU_KERNEL(float)
REGISTER_SLICE_GPU_KERNEL(double)
REGISTER_SLICE_GPU_KERNEL(int32_t)
REGISTER_SLICE_GPU_KERNEL(int64_t)
REGISTER_SLICE_GPU_KERNEL(int8_t)

}  // namespace oneflow
