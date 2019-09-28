#include "oneflow/core/kernel/gather_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/kernel.h"
#include <assert.h>

namespace oneflow {

namespace {

Shape GetFlatShape(const Shape& shape, int64_t axis) {
  CHECK_GT(shape.NumAxes(), 0);
  CHECK_GE(axis, 0);
  CHECK_LT(axis, shape.NumAxes());
  return Shape({shape.Count(0, axis), shape.At(axis), shape.Count(axis + 1)});
}

template<typename K>
__device__ int64_t GetInOffset(int64_t out_offset, const K* indices, int64_t num_indices,
                               int64_t gather_dim_size, int64_t inner_dim_size,
                               const int64_t offset) {
  const int64_t outer_dim_elem_cnt = num_indices * inner_dim_size;
  const int64_t outer_idx = out_offset / outer_dim_elem_cnt;
  const int64_t indices_idx = out_offset % outer_dim_elem_cnt / inner_dim_size;
  const int64_t inner_idx = out_offset % inner_dim_size;
  assert(indices[indices_idx] >= 0);
  const int64_t idx = indices[indices_idx] - offset;
  if (idx >= 0 && idx < gather_dim_size) {
    return outer_idx * gather_dim_size * inner_dim_size + idx * inner_dim_size + inner_idx;
  } else {
    return -1;
  }
}

template<typename T, typename K>
__global__ void GatherForwardGpu(int64_t elem_cnt, const K* indices, int64_t num_indices,
                                 const T* in, int64_t gather_dim_size, int64_t inner_dim_size,
                                 T* out, const int64_t offset) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int64_t in_offset =
        GetInOffset<K>(i, indices, num_indices, gather_dim_size, inner_dim_size, offset);
    if (in_offset < 0) {
      out[i] = 0;
    } else {
      out[i] = in[in_offset];
    }
  }
}

template<typename T, typename K>
__global__ void GatherBackwardGpu(int64_t elem_cnt, const K* indices, int64_t num_indices,
                                  const T* out_diff, int64_t gather_dim_size,
                                  int64_t inner_dim_size, T* in_diff, const int64_t offset) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int64_t in_offset =
        GetInOffset<K>(i, indices, num_indices, gather_dim_size, inner_dim_size, offset);
    if (in_offset >= 0) { gpu_atomic_add(in_diff + in_offset, out_diff[i]); }
  }
}

}  // namespace

template<typename T, typename K>
struct GatherKernelUtilImpl<DeviceType::kGPU, T, K> final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* in,
                      const Shape& flat_in_shape, T* out, const int64_t offset) {
    const int64_t elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
    GatherForwardGpu<T, K>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, indices, num_indices, in, flat_in_shape.At(1), flat_in_shape.At(2), out,
            offset);
  }
  static void Backward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* out_diff,
                       const Shape& flat_in_shape, T* in_diff, const int64_t offset) {
    const int64_t elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
    GatherBackwardGpu<T, K>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, indices, num_indices, out_diff, flat_in_shape.At(1), flat_in_shape.At(2),
            in_diff, offset);
  }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
template<typename K>
struct GatherKernelUtilImpl<DeviceType::kGPU, float16, K> final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const float16* in,
                      const Shape& flat_in_shape, float16* out, const int64_t offset) {
    const int64_t elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
    GatherForwardGpu<half, K>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, indices, num_indices, reinterpret_cast<const half*>(in), flat_in_shape.At(1),
            flat_in_shape.At(2), reinterpret_cast<half*>(out), offset);
  }
  static void Backward(DeviceCtx* ctx, const K* indices, int64_t num_indices,
                       const float16* out_diff, const Shape& flat_in_shape, float16* in_diff,
                       const int64_t offset) {
    const int64_t elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
    GatherBackwardGpu<half, K>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, indices, num_indices, reinterpret_cast<const half*>(out_diff),
            flat_in_shape.At(1), flat_in_shape.At(2), reinterpret_cast<half*>(in_diff), offset);
  }
};
#endif

#define INITIATE_GATHER_KERNEL_UTIL_GPU_IMPL(in_type_pair, index_type_pair)              \
  template struct GatherKernelUtilImpl<DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                       OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GATHER_KERNEL_UTIL_GPU_IMPL,
                                 FLOATING_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(int32_t,
                                                                             DataType::kInt32)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                                     FLOAT16_DATA_TYPE_SEQ
#endif
                                 ,
                                 INT_DATA_TYPE_SEQ);
#undef INITIATE_GATHER_KERNEL_UTIL_GPU_IMPL

class LocalGatherInt8GPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalGatherInt8GPUKernel);
  LocalGatherInt8GPUKernel() = default;
  ~LocalGatherInt8GPUKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().local_gather_conf();
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* indices = BnInOp2Blob("indices");
    CHECK_EQ(indices->data_type(), DataType::kInt32);
    const Blob* in = BnInOp2Blob("in");
    CHECK_EQ(in->data_type(), DataType::kInt8);
    Blob* out = BnInOp2Blob("out");
    const int64_t num_indices = indices->shape().elem_cnt();
    const Shape flat_in_shape =
        GetFlatShape(in->shape(), this->op_conf().local_gather_conf().axis());
    const int64_t elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
    GatherForwardGpu<int8_t, int32_t><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                        ctx.device_ctx->cuda_stream()>>>(
        elem_cnt, indices->dptr<int32_t>(), num_indices, in->dptr<int8_t>(), flat_in_shape.At(1),
        flat_in_shape.At(2), out->mut_dptr<int8_t>(), 0);
  }
};

template<typename T>
class LocalGatherGPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalGatherGPUKernel);
  LocalGatherGPUKernel() = default;
  ~LocalGatherGPUKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().local_gather_conf();
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* indices = BnInOp2Blob("indices");
    CHECK_EQ(indices->data_type(), DataType::kInt32);
    const Blob* in = BnInOp2Blob("in");
    Blob* out = BnInOp2Blob("out");
    const int64_t num_indices = indices->shape().elem_cnt();
    const Shape flat_in_shape =
        GetFlatShape(in->shape(), this->op_conf().local_gather_conf().axis());
    const int64_t elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
    GatherForwardGpu<T, int32_t><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                   ctx.device_ctx->cuda_stream()>>>(
        elem_cnt, indices->dptr<int32_t>(), num_indices, in->dptr<T>(), flat_in_shape.At(1),
        flat_in_shape.At(2), out->mut_dptr<T>(), 0);
  }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLocalGatherConf, DeviceType::kGPU, int8_t,
                                      LocalGatherInt8GPUKernel)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLocalGatherConf, DeviceType::kGPU, int32_t,
                                      LocalGatherGPUKernel<int32_t>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLocalGatherConf, DeviceType::kGPU, float,
                                      LocalGatherGPUKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLocalGatherConf, DeviceType::kGPU, double,
                                      LocalGatherGPUKernel<double>)

}  // namespace oneflow
