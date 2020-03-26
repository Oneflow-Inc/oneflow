#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/random_generator.h"
#include "oneflow/customized/kernels/radix_sort.cuh"

namespace oneflow {

namespace {

class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(const int32_t& batch_size, const int32_t& capacity, void* ptr)
      : capacity_{capacity},
        random_value_elem_cnt_{batch_size},
        sorted_value_elem_cnt_{batch_size},
        indices_elem_cnt_{batch_size} {
    const int32_t random_value_aligned_bytes =
        GetCudaAlignedSize(random_value_elem_cnt_ * sizeof(float));
    const int32_t sorted_value_aligned_bytes =
        GetCudaAlignedSize(sorted_value_elem_cnt_ * sizeof(float));
    const int32_t indices_aligned_bytes = GetCudaAlignedSize(indices_elem_cnt_ * sizeof(int32_t));
    random_value_ptr_ = reinterpret_cast<float*>(ptr);
    sorted_value_ptr_ = reinterpret_cast<float*>(reinterpret_cast<char*>(random_value_ptr_)
                                                 + random_value_aligned_bytes);
    indices_ptr_ = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(sorted_value_ptr_)
                                              + sorted_value_aligned_bytes);
    temp_storage_ptr_ =
        reinterpret_cast<void*>(reinterpret_cast<char*>(indices_ptr_) + indices_aligned_bytes);
    temp_storage_bytes_ =
        capacity_ - random_value_aligned_bytes - sorted_value_aligned_bytes - indices_aligned_bytes;
    CHECK_GE(temp_storage_bytes_, 0);
  }
  ~TmpBufferManager() = default;

  float* RandomValuePtr() const { return random_value_ptr_; }
  float* SortedValuePtr() const { return sorted_value_ptr_; }
  int32_t* IndicesPtr() const { return indices_ptr_; }
  void* TempStoragePtr() const { return temp_storage_ptr_; }

  int32_t RandomValueElemCnt() const { return random_value_elem_cnt_; }
  int32_t SortedValueElemCnt() const { return sorted_value_elem_cnt_; }
  int32_t IndicesElemCnt() const { return indices_elem_cnt_; }
  int32_t TempStorageBytes() const { return temp_storage_bytes_; }

 private:
  int32_t capacity_;

  float* random_value_ptr_;
  float* sorted_value_ptr_;
  int32_t* indices_ptr_;
  void* temp_storage_ptr_;

  int32_t random_value_elem_cnt_;
  int32_t sorted_value_elem_cnt_;
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
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t batch_size = y->shape().At(0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    TmpBufferManager buf_manager(batch_size, static_cast<int32_t>(tmp_buffer->shape().elem_cnt()),
                                 tmp_buffer->mut_dptr<void>());
    random_generator_->Uniform(batch_size, buf_manager.RandomValuePtr());
    InitializeIndices<<<BlocksNum4ThreadsNum(batch_size), kCudaThreadsNumPerBlock, 0,
                        ctx->device_ctx()->cuda_stream()>>>(batch_size, buf_manager.IndicesPtr());
    const int32_t argsort_instance_num = 1;
    const int32_t argsort_instance_size = batch_size;
    SortPairsAscending(buf_manager.RandomValuePtr(), buf_manager.IndicesPtr(), argsort_instance_num,
                       argsort_instance_size, buf_manager.TempStoragePtr(),
                       buf_manager.TempStorageBytes(), buf_manager.SortedValuePtr(),
                       y->mut_dptr<int32_t>(), ctx->device_ctx()->cuda_stream());
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
      const Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      const int32_t batch_size = y_shape->At(0);

      const int32_t random_value_aligned_bytes = GetCudaAlignedSize(batch_size * sizeof(float));
      const int32_t sorted_value_aligned_bytes = GetCudaAlignedSize(batch_size * sizeof(float));
      const int32_t indices_aligned_bytes = GetCudaAlignedSize(batch_size * sizeof(int32_t));
      const int32_t argsort_instance_num = 1;
      const int32_t argsort_instance_size = batch_size;
      const int32_t temp_storage_bytes = InferTempStorageForSortPairsAscending<float, int32_t>(
          argsort_instance_num, argsort_instance_size);

      return random_value_aligned_bytes + sorted_value_aligned_bytes + indices_aligned_bytes
             + temp_storage_bytes;
    });

}  // namespace oneflow
