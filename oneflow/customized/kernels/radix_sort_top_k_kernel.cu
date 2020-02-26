#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/radix_sort.cuh"

namespace oneflow {

namespace {

template<typename T>
class TmpBufManager final {
 public:
  TmpBufManager(int32_t tmp_buffer_bytes, char* tmp_buffer_ptr, const ShapeView& in_shape)
      : capacity_{tmp_buffer_bytes},
        sorted_in_bytes_{in_shape.elem_cnt() * sizeof(T)},
        indices_bytes_{in_shape.elem_cnt() * sizeof(int32_t)},
        sorted_indices_bytes_{in_shape.elem_cnt() * sizeof(int32_t)},
        temp_storage_bytes_{capacity_ - sorted_in_bytes_ - indices_bytes_ - sorted_indices_bytes_} {
    sorted_in_ptr_ = (T*)tmp_buffer_ptr;
    indices_ptr_ = (int32_t*)(tmp_buffer_ptr + sorted_in_bytes_);
    sorted_indices_ptr_ = indices_ptr_ + in_shape.elem_cnt();
    temp_storage_ptr_ = (void*)sorted_indices_ptr_ + sorted_indices_bytes_;
  }

  int32_t Capacity() const { return capacity_; }

  T* SortedInPtr() const { return sorted_in_ptr_; }
  int32_t* IndicesPtr() const { return indices_ptr_; }
  int32_t* SortedIndicesPtr() const { return sorted_indices_ptr_; }
  void* TempStoragePtr() const { return temp_storage_ptr_; }

  int32_t GetSortedInBytes() { return sorted_in_bytes_; }
  int32_t GetIndicesBytes() { return indices_bytes_; }
  int32_t GetSortedIndicesBytes() { return sorted_indices_bytes_; }
  int32_t GetTempStorageBytes() { return temp_storage_bytes_; }

 private:
  int32_t capacity_;

  T* sorted_in_ptr_;
  int32_t* indices_ptr_;
  int32_t* sorted_indices_ptr_;
  void* temp_storage_ptr_;

  int32_t sorted_in_bytes_;
  int32_t indices_bytes_;
  int32_t sorted_indices_bytes_;
  int32_t temp_storage_bytes_;
};

__global__ void RadixSortTopKInitializeKernel(int32_t* indices_ptr, int32_t instance_size) {
  for (int32_t i = threadIdx.x; i < instance_size; i += blockDim.x) {
    indices_ptr[blockIdx.x * instance_size + i] = i;
  }
}

__global__ void RadixSortTopKWriteToOutputKernel(const int32_t* sorted_indices_ptr,
                                                 int32_t instance_size, int32_t k,
                                                 int32_t* output_ptr) {
  for (int32_t i = threadIdx.x; i < k; i += blockDim.x) {
    output_ptr[blockIdx.x * k + i] = sorted_indices_ptr[blockIdx.x * instance_size + i];
  }
}

}  // namespace

template<typename T>
class RadixSortTopKKernel final : public user_op::OpKernel {
 public:
  RadixSortTopKKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  RadixSortTopKKernel() = default;
  ~RadixSortTopKKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    auto* buf_manager = new TmpBufManager<T>(static_cast<int32_t>(tmp_buffer->shape().elem_cnt()),
                                             tmp_buffer->mut_dptr<char>(), in->shape());

    const int32_t instance_size = in->shape().At(in->shape().NumAxes() - 1);
    const int32_t instance_num = in->shape().elem_cnt() / instance_size;
    const int32_t k = std::min(ctx->GetAttr<int32_t>("k"), instance_size);
    int32_t num_thread =
        instance_size <= kCudaThreadsNumPerBlock ? instance_size : kCudaThreadsNumPerBlock;
    RadixSortTopKInitializeKernel<<<instance_num, num_thread, 0,
                                    ctx->device_ctx()->cuda_stream()>>>(buf_manager->IndicesPtr(),
                                                                        instance_size);
    SortPairsDescending(in->dptr<T>(), buf_manager->IndicesPtr(), instance_num, instance_size,
                        buf_manager->TempStoragePtr(), buf_manager->GetTempStorageBytes(),
                        buf_manager->SortedInPtr(), buf_manager->SortedIndicesPtr(),
                        ctx->device_ctx()->cuda_stream());
    num_thread = k <= kCudaThreadsNumPerBlock ? k : kCudaThreadsNumPerBlock;
    RadixSortTopKWriteToOutputKernel<<<instance_num, num_thread, 0,
                                       ctx->device_ctx()->cuda_stream()>>>(
        buf_manager->SortedIndicesPtr(), instance_size, k, out->mut_dptr<int32_t>());
  };
};

#define REGISTER_RADIX_SORT_TOP_K_KERNEL(dtype)                                                  \
  REGISTER_USER_KERNEL("top_k")                                                                  \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                          \
        return new RadixSortTopKKernel<dtype>(ctx);                                              \
      })                                                                                         \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                      \
        const user_op::TensorDesc* in_desc = ctx.TensorDesc4ArgNameAndIndex("in", 0);            \
        return ctx.device() == DeviceType::kGPU && ctx.GetAttr<int32_t>("k") > 128               \
               && in_desc->data_type() == GetDataType<dtype>::value;                             \
      })                                                                                         \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                               \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);                             \
        const int32_t instance_size = in_shape->dim_vec().back();                                \
        return static_cast<int32_t>(in_shape->elem_cnt() * (sizeof(dtype) + 2 * sizeof(int32_t)) \
                                    + InferTempStorageForSortPairsDescending<dtype, int32_t>( \
                                          in_shape->elem_cnt() / instance_size, instance_size)); \
      });

REGISTER_RADIX_SORT_TOP_K_KERNEL(float)
REGISTER_RADIX_SORT_TOP_K_KERNEL(double)
REGISTER_RADIX_SORT_TOP_K_KERNEL(int32_t)
REGISTER_RADIX_SORT_TOP_K_KERNEL(int64_t)

}  // namespace oneflow
