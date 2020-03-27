#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/arg_where_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

constexpr int kFlatIndexToNdIndexProposedLaunchBlocks = 128;

template<typename T, size_t NDims>
struct StrideIterator {
  typedef StrideIterator self_type;
  typedef std::ptrdiff_t difference_type;
  typedef T value_type;
  typedef T* pointer;
  typedef T& reference;
  typedef std::random_access_iterator_tag iterator_category;

  explicit StrideIterator(T* ptr, size_t max_iters) : ptr_(ptr), max_iters_(max_iters) {}

  OF_DEVICE_FUNC reference operator[](int i) {
    assert(0 <= i && i < max_iters_);
    return *(ptr_ + (i * NDims));
  }

 private:
  T* ptr_;
  size_t max_iters_;
};

template<typename T, size_t NDims>
__global__ void CudaOffsetToNdIndexInplace(NdIndexOffsetHelper<T, NDims> index_converter,
                                           const T* num_indices_ptr, T* indices_ptr) {
  CUDA_1D_KERNEL_LOOP_T(T, i, *num_indices_ptr) {
    T* cur_indices_ptr = indices_ptr + i * NDims;
    index_converter.OffsetToNdIndex(*cur_indices_ptr, cur_indices_ptr);
  }
}

template<typename T>
struct IsTrue {
  OF_DEVICE_FUNC bool operator()(const T& val) const { return static_cast<bool>(val); }
};

template<typename T, typename I, typename Iter>
cudaError_t SelectTrue(cudaStream_t stream, int num_items, void* tmp, size_t& tmp_bytes,
                       const T* flags, Iter out_iter, I* num_selected) {
  IsTrue<T> is_true;
  cub::TransformInputIterator<bool, IsTrue<T>, const T*> flag_iter(flags, is_true);
  cub::CountingInputIterator<I> offset_counter(0);
  return cub::DeviceSelect::Flagged(tmp, tmp_bytes, offset_counter, flag_iter, out_iter,
                                    num_selected, num_items, stream, false);
}

}  // namespace

template<typename T, typename I>
cudaError_t InferSelectTrueTmpBufferSize(cudaStream_t stream, int num_items, size_t& tmp_bytes) {
  return SelectTrue<T, I, I*>(stream, num_items, nullptr, tmp_bytes, nullptr, nullptr, nullptr);
}

template<typename T, typename I, size_t NDims>
class ArgWhereGpuKernel : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ArgWhereGpuKernel);
  ArgWhereGpuKernel() = default;
  ~ArgWhereGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in = BnInOp2Blob("in");
    Blob* out = BnInOp2Blob("out");
    Blob* out_size = BnInOp2Blob("out_size");
    Blob* tmp = BnInOp2Blob("tmp");
    const int64_t elem_cnt = in->shape().elem_cnt();
    CHECK_LE(elem_cnt, std::numeric_limits<I>::max());
    size_t tmp_bytes = 0;
    CudaCheck(
        InferSelectTrueTmpBufferSize<T, I>(ctx.device_ctx->cuda_stream(), elem_cnt, tmp_bytes));
    CHECK_LE(tmp_bytes, tmp->shape().elem_cnt());

    if (NDims == 1) {
      CudaCheck(SelectTrue<T, I, I*>(ctx.device_ctx->cuda_stream(), elem_cnt, tmp->mut_dptr(),
                                     tmp_bytes, in->dptr<T>(), out->mut_dptr<I>(),
                                     out_size->mut_dptr<I>()));
    } else {
      StrideIterator<I, NDims> out_iter(out->mut_dptr<I>(), elem_cnt);
      CudaCheck(SelectTrue<T, I, StrideIterator<I, NDims>>(
          ctx.device_ctx->cuda_stream(), elem_cnt, tmp->mut_dptr(), tmp_bytes, in->dptr<T>(),
          out_iter, out_size->mut_dptr<I>()));

      fixed_vector<I, NDims> dims(NDims);
      std::transform(in->shape().ptr(), in->shape().ptr() + in->shape().NumAxes(), dims.begin(),
                     [](int64_t dim) { return static_cast<I>(dim); });
      NdIndexOffsetHelper<I, NDims> index_converter(dims.data(), dims.size());
      CudaOffsetToNdIndexInplace<I, NDims>
          <<<kFlatIndexToNdIndexProposedLaunchBlocks, kCudaThreadsNumPerBlock, 0,
             ctx.device_ctx->cuda_stream()>>>(index_converter, out_size->dptr<I>(),
                                              out->mut_dptr<I>());
    }
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_INFER_SELECT_TRUE_TMP_BUFFER_SIZE,
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

#define REGISTER_ARG_WHERE_GPU_KERNELS(dtype_pair, itype_pair)             \
  REGISTER_ARG_WHERE_KERNELS_AT_NDIMS(ArgWhereGpuKernel, DeviceType::kGPU, \
                                      OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(itype_pair))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ARG_WHERE_GPU_KERNELS, ARITHMETIC_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
