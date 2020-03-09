#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/random_generator.h"
#include "oneflow/customized/kernels/radix_sort.cuh"

namespace oneflow {

namespace {

class TmpBufferManager final {
 public:
  TmpBufferManager(int32_t capacity, void* ptr, const ShapeView& in_shape)
      : capacity_{capacity},
        sorted_in_elem_cnt_{in_shape.At(0)},
        indices_elem_cnt_{sorted_in_elem_cnt_} {
    const int32_t in_aligned_bytes = GetCudaAlignedSize(sorted_in_elem_cnt_ * sizeof(float));
    const int32_t indices_aligned_bytes = GetCudaAlignedSize(indices_elem_cnt_ * sizeof(int32_t));
    in_ptr_ = reinterpret_cast<float*>(ptr);
    sorted_in_ptr_ = in_ptr_ + in_aligned_bytes;
    indices_ptr_ =
        reinterpret_cast<int32_t*>(reinterpret_cast<char*>(sorted_in_ptr_) + in_aligned_bytes);
    temp_storage_ptr_ =
        reinterpret_cast<void*>(reinterpret_cast<char*>(indices_ptr_) + indices_aligned_bytes);
    temp_storage_bytes_ = capacity_ - in_aligned_bytes * 2 - indices_aligned_bytes;
    CHECK_GE(temp_storage_bytes_, 0);
  }
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  ~TmpBufferManager() = default;

  float* InPtr() const { return sorted_in_ptr_; }
  float* SortedInPtr() const { return sorted_in_ptr_; }
  int32_t* IndicesPtr() const { return indices_ptr_; }
  void* TempStoragePtr() const { return temp_storage_ptr_; }

  int32_t SortedInElemCnt() const { return sorted_in_elem_cnt_; }
  int32_t IndicesElemCnt() const { return indices_elem_cnt_; }
  int32_t TempStorageBytes() const { return temp_storage_bytes_; }

 private:
  int32_t capacity_;

  float* in_ptr_;
  float* sorted_in_ptr_;
  int32_t* indices_ptr_;
  void* temp_storage_ptr_;

  int32_t sorted_in_elem_cnt_;
  int32_t indices_elem_cnt_;
  int32_t temp_storage_bytes_;
};

__global__ void InitializeIndices(int32_t elem_cnt, int32_t* indices_ptr, int32_t instance_size) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { indices_ptr[i] = i % instance_size; };
}

}  // namespace

template<DeviceType device_type, typename T>
class RandomBatchPermutationKernel final : public user_op::OpKernel {
 public:
  RandomBatchPermutationKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  RandomBatchPermutationKernel() = default;
  ~RandomBatchPermutationKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    // TODO: tsai, chengcheng do the initialization in kernel init when interface ready
    if (random_generator_.get() == nullptr) {
      int64_t seed = GetCurTime();
      const bool has_seed = ctx->GetAttr<int32_t>("has_seed") == 1;
      if (has_seed) { seed = ctx->GetAttr<int64_t>("seed"); }
      random_generator_.reset(new RandomGenerator<DeviceType::kGPU>(seed, ctx->device_ctx()));
    }
    user_op::Tensor* like = ctx->Tensor4ArgNameAndIndex("like", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    TmpBufferManager buf_manager(static_cast<int32_t>(tmp_buffer->shape().elem_cnt()),
                                 tmp_buffer->mut_dptr<void>(), like->shape());
    random_generator_->Uniform(like->shape().At(0), buf_manager.InPtr());
    const int32_t elem_cnt = like->shape().At(0);
    const int32_t instance_size = 1;
    const int32_t instance_num = like->shape().At(0);
    InitializeIndices<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                        ctx->device_ctx()->cuda_stream()>>>(elem_cnt, buf_manager.IndicesPtr(),
                                                            instance_size);
    SortPairsAscending(buf_manager.InPtr(), buf_manager.IndicesPtr(), instance_num, instance_size,
                       buf_manager.TempStoragePtr(), buf_manager.TempStorageBytes(),
                       buf_manager.SortedInPtr(), out->mut_dptr<int32_t>(),
                       ctx->device_ctx()->cuda_stream());
  };

  std::unique_ptr<RandomGenerator<DeviceType::kGPU>> random_generator_;
};

#define REGISTER_RANDOM_BATCH_PERMUTATION_KERNEL(dtype, dev)                                   \
  REGISTER_USER_KERNEL("generate_random_batch_permutation_indices")                            \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                        \
        return new RandomBatchPermutationKernel<dev, dtype>(ctx);                              \
      })                                                                                       \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);        \
        return ctx.device_type() == dev && out_desc->data_type() == GetDataType<dtype>::value; \
      });

REGISTER_RANDOM_BATCH_PERMUTATION_KERNEL(int32_t, DeviceType::kGPU)
REGISTER_RANDOM_BATCH_PERMUTATION_KERNEL(int64_t, DeviceType::kGPU)
REGISTER_RANDOM_BATCH_PERMUTATION_KERNEL(float, DeviceType::kGPU)
REGISTER_RANDOM_BATCH_PERMUTATION_KERNEL(double, DeviceType::kGPU)

}  // namespace oneflow
