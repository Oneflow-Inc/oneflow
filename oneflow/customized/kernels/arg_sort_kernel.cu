#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/customized/kernels/gpu_sort_utils/radix_sort.cuh"

namespace oneflow {

namespace {

template<typename T>
class TmpBufManager final {
 public:
  TmpBufManager(int32_t tmp_buffer_bytes, char* tmp_buffer_ptr, const ShapeView& in_shape)
      : capacity_{tmp_buffer_bytes},
        sorted_in_bytes_{in_shape.elem_cnt() * sizeof(T)},
        indices_bytes_{in_shape.elem_cnt() * sizeof(int32_t)},
        temp_storage_bytes_{capacity_ - sorted_in_bytes_ - indices_bytes_} {
    sorted_in_ptr_ = (T*)tmp_buffer_ptr;
    indices_ptr_ = (int32_t*)(tmp_buffer_ptr + sorted_in_bytes_);
    temp_storage_ptr_ = (void*)((char*)indices_ptr_ + indices_bytes_);
  }

  int32_t Capacity() const { return capacity_; }

  T* SortedInPtr() const { return sorted_in_ptr_; }
  int32_t* IndicesPtr() const { return indices_ptr_; }
  void* TempStoragePtr() const { return temp_storage_ptr_; }

  int32_t GetSortedInBytes() { return sorted_in_bytes_; }
  int32_t GetIndicesBytes() { return indices_bytes_; }
  int32_t GetTempStorageBytes() { return temp_storage_bytes_; }

 private:
  int32_t capacity_;

  T* sorted_in_ptr_;
  int32_t* indices_ptr_;
  int32_t* sorted_indices_ptr_;
  void* temp_storage_ptr_;

  int32_t sorted_in_bytes_;
  int32_t indices_bytes_;
  int32_t temp_storage_bytes_;
};

__global__ void RadixSortTopKInitializeKernel(int32_t* indices_ptr, int32_t instance_size) {
  for (int32_t i = threadIdx.x; i < instance_size; i += blockDim.x) {
    indices_ptr[blockIdx.x * instance_size + i] = i;
  }
}

}  // namespace

template<typename T>
class GpuArgSortKernel final : public user_op::OpKernel {
 public:
  GpuArgSortKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  GpuArgSortKernel() = default;
  ~GpuArgSortKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const int32_t instance_size = in->shape().At(in->shape().NumAxes() - 1);
    const int32_t instance_num = in->shape().elem_cnt() / instance_size;
    auto* buf_manager = new TmpBufManager<T>(static_cast<int32_t>(tmp_buffer->shape().elem_cnt()),
                                             tmp_buffer->mut_dptr<char>(), in->shape());
    const std::string& dir = ctx->GetAttr<std::string>("dir");

    const int32_t num_thread =
        instance_size <= kCudaThreadsNumPerBlock ? instance_size : kCudaThreadsNumPerBlock;
    RadixSortTopKInitializeKernel<<<instance_num, num_thread, 0,
                                    ctx->device_ctx()->cuda_stream()>>>(buf_manager->IndicesPtr(),
                                                                        instance_size);
    if (dir == "ASCENDING") {
      SortPairsAscending(in->dptr<T>(), buf_manager->IndicesPtr(), instance_num, instance_size,
                         buf_manager->TempStoragePtr(), buf_manager->GetTempStorageBytes(),
                         buf_manager->SortedInPtr(), out->mut_dptr<int32_t>(),
                         ctx->device_ctx()->cuda_stream());
    } else if (dir == "DESCENDING") {
      SortPairsDescending(in->dptr<T>(), buf_manager->IndicesPtr(), instance_num, instance_size,
                          buf_manager->TempStoragePtr(), buf_manager->GetTempStorageBytes(),
                          buf_manager->SortedInPtr(), out->mut_dptr<int32_t>(),
                          ctx->device_ctx()->cuda_stream());
    } else {
      UNIMPLEMENTED();
    }
  };
};

#define REGISTER_GPU_ARG_SORT_KERNEL(dtype)                                                  \
  REGISTER_USER_KERNEL("arg_sort")                                                           \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                      \
        return new GpuArgSortKernel<dtype>(ctx);                                             \
      })                                                                                     \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* in_desc = ctx.TensorDesc4ArgNameAndIndex("in", 0);        \
        return ctx.device() == DeviceType::kGPU                                              \
               && in_desc->data_type() == GetDataType<dtype>::value;                         \
      })                                                                                     \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                           \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);                         \
        const int32_t instance_size = in_shape->dim_vec().back();                            \
        const int32_t instance_num = in_shape->elem_cnt() / instance_size;                   \
        int32_t temp_storage_bytes = -1;                                                     \
        const std::string& dir = ctx->GetAttr<std::string>("dir");                           \
        if (dir == "ASCENDING") {                                                            \
          temp_storage_bytes = InferTempStorageForSortingPairsAscending<dtype, int32_t>(     \
              instance_num, instance_size);                                                  \
        } else if (dir == "DESCENDING") {                                                    \
          temp_storage_bytes = InferTempStorageForSortingPairsDescending<dtype, int32_t>(    \
              instance_num, instance_size);                                                  \
        } else {                                                                             \
          UNIMPLEMENTED();                                                                   \
        }                                                                                    \
        return static_cast<int32_t>(in_shape->elem_cnt() * (sizeof(dtype) + sizeof(int32_t)) \
                                    + temp_storage_bytes);                                   \
      });

REGISTER_GPU_ARG_SORT_KERNEL(float)
REGISTER_GPU_ARG_SORT_KERNEL(double)
REGISTER_GPU_ARG_SORT_KERNEL(int8_t)
REGISTER_GPU_ARG_SORT_KERNEL(int32_t)
REGISTER_GPU_ARG_SORT_KERNEL(int64_t)

}  // namespace oneflow
