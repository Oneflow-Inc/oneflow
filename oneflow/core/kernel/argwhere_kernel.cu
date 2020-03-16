#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/argwhere_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

constexpr int kFlatIndexToNdIndexProposedLaunchBlocks = 128;

template<typename T, size_t NDims>
struct NDimensionIterator {
  typedef NDimensionIterator self_type;
  typedef std::ptrdiff_t difference_type;
  typedef T value_type;
  typedef T* pointer;
  typedef T& reference;
  typedef std::random_access_iterator_tag iterator_category;

  explicit NDimensionIterator(T* ptr, size_t max_iters) : ptr_(ptr), max_iters_(max_iters) {}

  OF_DEVICE_FUNC reference operator[](int i) {
    assert(0 <= i && i < max_iters_);
    return *(ptr_ + (i * NDims));
  }

 private:
  T* ptr_;
  size_t max_iters_;
};

template<typename T, size_t NDims>
__global__ void CudaFlatIndexToNdIndex(const int64_t* num_indices_ptr,
                                       NdIndexOffsetHelper<T, NDims> index_convertor, T* out_ptr) {
  int32_t num_indices = static_cast<int32_t>(__ldg(num_indices_ptr));
  CUDA_1D_KERNEL_LOOP(i, num_indices) {
    T* cur_indices_ptr = out_ptr + i * NDims;
    OffsetToNdIndex(__ldg(cur_indices_ptr), cur_indices_ptr);
  }
}

template<typename T>
struct IsNonzero {
  OF_DEVICE_FUNC bool operator()(const T& val) const { return (val != static_cast<T>(0)); }
};

}  // namespace

template<typename T, typename I>
cudaError_t InferCubSelectFlaggedTempStorageBytes(DeviceCtx* ctx, int num_items,
                                                  size_t& tmp_bytes) {
  return CubSelectFlagged<T, I, I*>(ctx ? ctx->cuda_stream() : 0, num_items, nullptr, tmp_bytes,
                                    nullptr, nullptr, nullptr)
}

template<typename T, typename I, typename Iter>
cudaError_t CubSelectFlagged(cudaStream_t stream, int num_items, void* tmp, size_t& tmp_bytes,
                             const T* flags, Iter out_iter, int64_t* num_selected) {
  IsNonzero<T> is_nonzero;
  cub::TransformInputIterator<bool, IsNonzero<T>, const T*> flag_iter(flags, is_nonzero);
  cub::CountingInputIterator<I> flat_index_counter(0);
  return cub::DeviceSelect::Flagged(tmp, tmp_bytes, flat_index_counter, flag_iter, out_iter,
                                    num_selected, num_items, stream, false);
}

template<typename T, typename I, size_t NDims>
class ArgwhereGpuKernel : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ArgwhereGpuKernel);
  ArgwhereGpuKernel() = default;
  ~ArgwhereGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in = BnInOp2Blob("in");
    Blob* out = BnInOp2Blob("out");
    Blob* out_size = BnInOp2Blob("out_size");
    Blob* tmp = BnInOp2Blob("tmp");
    const int64_t elem_cnt = in->shape().elem_cnt();
    size_t tmp_bytes = 0;
    CudaCheck(InferCubSelectFlaggedTempStorageBytes<T, I>(ctx.device_ctx, elem_cnt, tmp_bytes));
    CHECK_LE(tmp_bytes, tmp->shape().elem_cnt());

    if (NDims == 1) {
      CudaCheck(CubSelectFlagged<T, I, I*>(ctx.device_ctx->cuda_stream(), elem_cnt, tmp->mut_dptr(),
                                           tmp_bytes, in->dptr<T>(), out->mut_dptr<I>(),
                                           out_size->mut_dptr<int64_t>()));
    } else {
      NDimensionIterator<I, NDims> out_iter(out->mut_dptr<I>(), elem_cnt);
      CudaCheck(CubSelectFlagged<T, I, NDimensionIterator<I, NDims>>(
          ctx.device_ctx->cuda_stream(), elem_cnt, tmp->mut_dptr(), tmp_bytes, in->dptr<T>(),
          out_iter, out_size->mut_dptr<int64_t>()));

      NdIndexOffsetHelper<I, NDims> index_convertor(in->shape().ptr());
      CudaFlatIndexToNdIndex<I, NDims>
          <<<kFlatIndexToNdIndexProposedLaunchBlocks, kCudaThreadsNumPerBlock, 0,
             ctx.device_ctx->cuda_stream()>>>(out_size->dptr<int64_t>(), index_convertor,
                                              out->mut_dptr<I>());
    }
  }
};

#define REGISTER_ARGWHERE_GPU_KERNEL_NDIMS(dtype, itype, ndims)                            \
  NEW_REGISTER_KERNEL(OperatorConf::kArgwhereConf, ArgwhereGpuKernel<dtype, itype, ndims>) \
      .SetIsMatchedPred([](const KernelConf& conf) {                                       \
        return (DeviceType::kGPU == conf.op_attribute().op_conf().device_type())           \
               && (GetDataType<dtype>::value == conf.argwhere_gpu_conf().data_type())      \
               && (GetDataType<itype>::value == conf.argwhere_gpu_conf().index_type())     \
               && (ndims == conf.argwhere_gpu_conf().num_axes());                          \
      });

#define REGISTER_ARGWHERE_GPU_KERNEL(dtype, itype)     \
  REGISTER_ARGWHERE_GPU_KERNEL_NDIMS(dtype, itype, 1); \
  REGISTER_ARGWHERE_GPU_KERNEL_NDIMS(dtype, itype, 2); \
  REGISTER_ARGWHERE_GPU_KERNEL_NDIMS(dtype, itype, 3); \
  REGISTER_ARGWHERE_GPU_KERNEL_NDIMS(dtype, itype, 4);

REGISTER_ARGWHERE_GPU_KERNEL(float, int32_t);
REGISTER_ARGWHERE_GPU_KERNEL(int8_t, int32_t);
REGISTER_ARGWHERE_GPU_KERNEL(int32_t, int32_t);

}  // namespace oneflow
