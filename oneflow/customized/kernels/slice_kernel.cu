#include "oneflow/core/framework/framework.h"

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

int64_t FixSliceBegin(int64_t begin, int64_t dims) {
  begin = (begin >= 0) ? begin : begin + dims;
  CHECK_GE(begin, 0);
  CHECK_LT(begin, dims);
  return begin;
}

int64_t FixSliceEnd(int64_t end, int64_t dims) {
  end = end >= 0 ? end : end + dims;
  CHECK_GT(end, 0);
  return std::min(end, dims);
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
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto begin_vec = ctx->GetAttr<std::vector<int64_t>>("begin");
    auto end_vec = ctx->GetAttr<std::vector<int64_t>>("end");
    auto stride_vec = ctx->GetAttr<std::vector<int64_t>>("stride");

    CHECK_LE(input->shape().NumAxes(), kSliceMaxDims);
    CHECK_EQ(input->shape().NumAxes(), output->shape().NumAxes());
    CHECK_EQ(input->shape().NumAxes(), begin_vec.size() + 1);
    CHECK_EQ(input->shape().NumAxes(), end_vec.size() + 1);
    CHECK_EQ(input->shape().NumAxes(), stride_vec.size() + 1);

    SliceGpuParams params;
    std::memset(&params, 0, sizeof(SliceGpuParams));
    params.ndims = input->shape().NumAxes();
    FOR_RANGE(int64_t, i, 0, params.ndims) {
      params.dims[i] = input->shape().At(i);
      params.sliced_dims[i] = output->shape().At(i);
      if (i == 0) {
        params.begin[i] = 0;
        params.end[i] = input->shape().At(i);
        params.stride[i] = 1;
      } else {
        params.begin[i] = FixSliceBegin(begin_vec.at(i - 1), params.dims[i]);
        params.end[i] = FixSliceEnd(end_vec.at(i - 1), params.dims[i]);
        params.stride[i] = stride_vec.at(i - 1);
      }
    }
    const int64_t elem_cnt = output->shape().elem_cnt();
    SliceForwardGpu<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                         ctx->device_ctx()->cuda_stream()>>>(elem_cnt, params, input->dptr<T>(),
                                                             output->mut_dptr<T>());
  }
};

}  // namespace oneflow
