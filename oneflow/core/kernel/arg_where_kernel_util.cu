#include "oneflow/core/kernel/arg_where_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/common/fixed_vector.h"
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

template<typename T, typename I>
size_t GetSelectTrueTempStorageSize(cudaStream_t stream, int num_items) {
  size_t tmp_bytes;
  CudaCheck(SelectTrue<T, I, I*>(stream, num_items, nullptr, tmp_bytes, nullptr, nullptr, nullptr));
  return tmp_bytes;
}

}  // namespace

template<typename T, typename I, size_t NDims>
struct ArgWhereForward<DeviceType::kGPU, T, I, NDims> {
  void operator()(DeviceCtx* ctx, const ShapeView& in_shape, const T* in_ptr, void* tmp,
                  size_t tmp_max_bytes, I* out_ptr, I* out_size_ptr) {
    CHECK_LE(in_shape.elem_cnt(), std::numeric_limits<I>::max());
    size_t tmp_bytes = ArgWhereWorkspace<DeviceType::kGPU, T, I>()(ctx, in_shape.elem_cnt());
    CHECK_LE(tmp_bytes, tmp_max_bytes);

    if (NDims == 1) {
      CudaCheck(SelectTrue<T, I, I*>(ctx->cuda_stream(), in_shape.elem_cnt(), tmp, tmp_bytes,
                                     in_ptr, out_ptr, out_size_ptr));
    } else {
      StrideIterator<I, NDims> out_iter(out_ptr, in_shape.elem_cnt());
      CudaCheck(SelectTrue<T, I, StrideIterator<I, NDims>>(
          ctx->cuda_stream(), in_shape.elem_cnt(), tmp, tmp_bytes, in_ptr, out_iter, out_size_ptr));

      fixed_vector<I, NDims> dims(NDims);
      std::transform(in_shape.ptr(), in_shape.ptr() + in_shape.NumAxes(), dims.begin(),
                     [](int64_t dim) { return static_cast<I>(dim); });
      NdIndexOffsetHelper<I, NDims> index_converter(dims.data(), dims.size());
      CudaOffsetToNdIndexInplace<I, NDims>
          <<<kFlatIndexToNdIndexProposedLaunchBlocks, kCudaThreadsNumPerBlock, 0,
             ctx->cuda_stream()>>>(index_converter, out_size_ptr, out_ptr);
    }
  }
};

template<typename T, typename I>
struct ArgWhereWorkspace<DeviceType::kGPU, T, I> {
  size_t operator()(DeviceCtx* ctx, int64_t n) {
    return GetSelectTrueTempStorageSize<T, I>(ctx ? ctx->cuda_stream() : 0, n);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ARG_WHERE_KERNEL_UTIL, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
