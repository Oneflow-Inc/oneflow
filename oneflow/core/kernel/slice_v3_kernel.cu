#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

constexpr size_t kSliceMaxDims = 8;

struct SliceGpuParams {
  int64_t ndims;
  int64_t entire_dims[kSliceMaxDims];
  int64_t part_dims[kSliceMaxDims];
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
    OffsetToDimCoords(i, params.ndims, params.part_dims, dim_coords);
#pragma unroll
    for (int64_t i = 0; i < params.ndims; ++i) {
      dim_coords[i] = params.begin[i] + params.stride[i] * dim_coords[i];
      assert(dim_coords[i] < params.end[i]);
    }
    int64_t offset = DimCoordsToOffset(params.ndims, params.entire_dims, dim_coords);
    part[i] = entire[offset];
  }
}

template<typename T>
__global__ void SliceBackwardGpu(const int n, SliceGpuParams params, const T* part, T* entire) {
  int64_t dim_coords[kSliceMaxDims];
  CUDA_1D_KERNEL_LOOP(i, n) {
    OffsetToDimCoords(i, params.ndims, params.part_dims, dim_coords);
#pragma unroll
    for (int64_t i = 0; i < params.ndims; ++i) {
      dim_coords[i] = params.begin[i] + params.stride[i] * dim_coords[i];
      assert(dim_coords[i] < params.end[i]);
    }
    int64_t offset = DimCoordsToOffset(params.ndims, params.entire_dims, dim_coords);
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
class SliceV3GpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceV3GpuKernel);
  SliceV3GpuKernel() = default;
  ~SliceV3GpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    const ShapeView& in_shape = in_blob->shape();
    const ShapeView& out_shape = out_blob->shape();
    const auto& conf = this->op_conf().slice_v3_conf();

    CHECK_LE(in_shape.NumAxes(), kSliceMaxDims);
    // DimSliceConf size is 1 less than shape's num of axes,
    // because Slice now don't support dim0 slice,
    CHECK_EQ(in_shape.NumAxes(), conf.dim_slice_conf_size() + 1);
    CHECK_EQ(in_shape.NumAxes(), out_shape.NumAxes());

    SliceGpuParams params;
    std::memset(&params, 0, sizeof(SliceGpuParams));
    params.ndims = in_shape.NumAxes();
    FOR_RANGE(int64_t, i, 0, params.ndims) {
      params.entire_dims[i] = in_shape.At(i);
      params.part_dims[i] = out_shape.At(i);
      if (i == 0) {
        params.begin[i] = 0;
        params.end[i] = in_shape.At(i);
        params.stride[i] = 1;
      } else {
        params.begin[i] = FixSliceBegin(conf.dim_slice_conf(i - 1).start(), params.entire_dims[i]);
        params.end[i] = FixSliceEnd(conf.dim_slice_conf(i - 1).end(), params.entire_dims[i]);
        params.stride[i] = conf.dim_slice_conf(i - 1).stride();
      }
    }
    const int64_t elem_cnt = out_blob->shape().elem_cnt();
    SliceForwardGpu<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                         ctx.device_ctx->cuda_stream()>>>(elem_cnt, params, in_blob->dptr<T>(),
                                                          out_blob->mut_dptr<T>());
  }
};

template<typename T>
class SliceGradV3GpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceGradV3GpuKernel);
  SliceGradV3GpuKernel() = default;
  ~SliceGradV3GpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* dy_blob = BnInOp2Blob("dy");
    Blob* dx_blob = BnInOp2Blob("dx");
    const ShapeView& dy_shape = dy_blob->shape();
    const ShapeView& dx_shape = dx_blob->shape();
    const auto& conf = this->op_conf().slice_grad_v3_conf();

    Memset<DeviceType::kGPU>(ctx.device_ctx, dx_blob->mut_dptr<T>(), 0,
                             dx_blob->ByteSizeOfBlobBody());

    CHECK_LE(dy_shape.NumAxes(), kSliceMaxDims);
    // DimSliceConf size is 1 less than shape's num of axes,
    // because Slice now don't support dim0 slice,
    CHECK_EQ(dy_shape.NumAxes(), conf.dim_slice_conf_size() + 1);
    CHECK_EQ(dy_shape.NumAxes(), dx_shape.NumAxes());

    SliceGpuParams params;
    std::memset(&params, 0, sizeof(SliceGpuParams));
    params.ndims = dy_shape.NumAxes();
    FOR_RANGE(int64_t, i, 0, params.ndims) {
      params.entire_dims[i] = dx_shape.At(i);
      params.part_dims[i] = dy_shape.At(i);
      if (i == 0) {
        params.begin[i] = 0;
        params.end[i] = dx_shape.At(i);
        params.stride[i] = 1;
      } else {
        params.begin[i] = FixSliceBegin(conf.dim_slice_conf(i - 1).start(), params.entire_dims[i]);
        params.end[i] = FixSliceEnd(conf.dim_slice_conf(i - 1).end(), params.entire_dims[i]);
        params.stride[i] = conf.dim_slice_conf(i - 1).stride();
      }
    }
    const int64_t elem_cnt = dy_blob->shape().elem_cnt();
    SliceBackwardGpu<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                          ctx.device_ctx->cuda_stream()>>>(elem_cnt, params, dy_blob->dptr<T>(),
                                                           dx_blob->mut_dptr<T>());
  }
};

#define REGISTER_SLICE_V3_GPU_KERNEL(dtype)                                                      \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSliceV3Conf, DeviceType::kGPU, dtype,     \
                                        SliceV3GpuKernel<dtype>)                                 \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSliceGradV3Conf, DeviceType::kGPU, dtype, \
                                        SliceGradV3GpuKernel<dtype>)

REGISTER_SLICE_V3_GPU_KERNEL(float);
REGISTER_SLICE_V3_GPU_KERNEL(double);
REGISTER_SLICE_V3_GPU_KERNEL(int8_t);
REGISTER_SLICE_V3_GPU_KERNEL(int32_t);
REGISTER_SLICE_V3_GPU_KERNEL(int64_t);

}  // namespace oneflow
