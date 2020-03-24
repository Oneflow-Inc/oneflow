#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/random_generator.h"
#include "oneflow/customized/kernels/radix_sort.cuh"

namespace oneflow {

namespace {

class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(int32_t capacity, void* ptr, const ShapeView& in_shape)
      : capacity_{capacity},
        in_elem_cnt_{in_shape.At(0)},
        sorted_in_elem_cnt_{in_elem_cnt_},
        indices_elem_cnt_{in_elem_cnt_} {
    const int32_t in_aligned_bytes = GetCudaAlignedSize(in_elem_cnt_ * sizeof(float));
    const int32_t sorted_in_aligned_bytes = GetCudaAlignedSize(sorted_in_elem_cnt_ * sizeof(float));
    const int32_t indices_aligned_bytes = GetCudaAlignedSize(indices_elem_cnt_ * sizeof(int32_t));
    in_ptr_ = reinterpret_cast<float*>(ptr);
    sorted_in_ptr_ = reinterpret_cast<float*>(reinterpret_cast<char*>(in_ptr_) + in_aligned_bytes);
    indices_ptr_ = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(sorted_in_ptr_)
                                              + sorted_in_aligned_bytes);
    temp_storage_ptr_ =
        reinterpret_cast<void*>(reinterpret_cast<char*>(indices_ptr_) + indices_aligned_bytes);
    temp_storage_bytes_ =
        capacity_ - in_aligned_bytes - sorted_in_aligned_bytes - indices_aligned_bytes;
    CHECK_GE(temp_storage_bytes_, 0);
  }
  ~TmpBufferManager() = default;

  float* InPtr() const { return in_ptr_; }
  float* SortedInPtr() const { return sorted_in_ptr_; }
  int32_t* IndicesPtr() const { return indices_ptr_; }
  void* TempStoragePtr() const { return temp_storage_ptr_; }

  int32_t InElemCnt() const { return in_elem_cnt_; }
  int32_t SortedInElemCnt() const { return sorted_in_elem_cnt_; }
  int32_t IndicesElemCnt() const { return indices_elem_cnt_; }
  int32_t TempStorageBytes() const { return temp_storage_bytes_; }

 private:
  int32_t capacity_;

  float* in_ptr_;
  float* sorted_in_ptr_;
  int32_t* indices_ptr_;
  void* temp_storage_ptr_;

  int32_t in_elem_cnt_;
  int32_t sorted_in_elem_cnt_;
  int32_t indices_elem_cnt_;
  int32_t temp_storage_bytes_;
};

__global__ void InitializeIndices(int32_t elem_cnt, int32_t* indices_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { indices_ptr[i] = i; };
}

}  // namespace

class GenerateRandomBatchPermutationIndicesGPUKernel final : public user_op::OpKernel {
 public:
  GenerateRandomBatchPermutationIndicesGPUKernel(user_op::KernelInitContext* ctx)
      : user_op::OpKernel(ctx) {
    int64_t seed = ctx->GetAttr<int64_t>("seed");
    random_generator_.reset(new RandomGenerator<DeviceType::kGPU>(seed, ctx->device_ctx()));
  }

  GenerateRandomBatchPermutationIndicesGPUKernel() = default;
  ~GenerateRandomBatchPermutationIndicesGPUKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    TmpBufferManager buf_manager(static_cast<int32_t>(tmp_buffer->shape().elem_cnt()),
                                 tmp_buffer->mut_dptr<void>(), x->shape());
    const int32_t elem_cnt = x->shape().At(0);
    const int32_t instance_size = x->shape().At(0);
    const int32_t instance_num = 1;
    random_generator_->Uniform(elem_cnt, buf_manager.InPtr());
    InitializeIndices<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                        ctx->device_ctx()->cuda_stream()>>>(elem_cnt, buf_manager.IndicesPtr());
    SortPairsAscending(buf_manager.InPtr(), buf_manager.IndicesPtr(), instance_num, instance_size,
                       buf_manager.TempStoragePtr(), buf_manager.TempStorageBytes(),
                       buf_manager.SortedInPtr(), y->mut_dptr<int32_t>(),
                       ctx->device_ctx()->cuda_stream());
  };

  std::unique_ptr<RandomGenerator<DeviceType::kGPU>> random_generator_;
};

REGISTER_USER_KERNEL("generate_random_batch_permutation_indices")
    .SetCreateFn([](oneflow::user_op::KernelInitContext* ctx) {
      return new GenerateRandomBatchPermutationIndicesGPUKernel(ctx);
    })
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {
      return ctx.device_type() == DeviceType::kGPU;
    })
    .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      const int32_t elem_cnt = x_shape->At(0);
      const int32_t instance_size = x_shape->At(0);
      const int32_t instance_num = 1;

      const int32_t in_aligned_bytes = GetCudaAlignedSize(elem_cnt * sizeof(float));
      const int32_t sorted_in_aligned_bytes = GetCudaAlignedSize(elem_cnt * sizeof(float));
      const int32_t indices_aligned_bytes = GetCudaAlignedSize(elem_cnt * sizeof(int32_t));
      const int32_t temp_storage_bytes =
          InferTempStorageForSortPairsAscending<float, int32_t>(instance_num, instance_size);

      return in_aligned_bytes + sorted_in_aligned_bytes + indices_aligned_bytes
             + temp_storage_bytes;
    });

}  // namespace oneflow
