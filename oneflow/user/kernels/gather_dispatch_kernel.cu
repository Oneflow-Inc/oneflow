/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/unique_op_util.h"
#include "oneflow/core/kernel/unique_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename K, int32_t N>
struct Param {
  const T* in;
  const K* in_size;
  T* out[N];
  K* count[N];
  int64_t range_start;
  int64_t num_out;
};

template<typename T, typename K>
__global__ void GetPartionBoundIndex(const int64_t n, const int64_t parallel_num,
                                     const int64_t num_classes_per_rank, const T* in_ptr,
                                     const K* in_size_ptr, K* out_ptr) {
  const K num = in_size_ptr[0];
  CUDA_1D_KERNEL_LOOP(i, num) {
    if (i != 0) {
      const T cur_in = in_ptr[i] / num_classes_per_rank;
      const T pre_in = in_ptr[i - 1] / num_classes_per_rank;
      if (cur_in > pre_in) {
        assert(cur_in < parallel_num);
#pragma unroll
        for (int32_t j = pre_in + 1; j <= cur_in; ++j) { out_ptr[j] = static_cast<K>(i); }
      }
    }
  }
  CUDA_1D_KERNEL_LOOP(i, parallel_num + 1) {
    const K first_in = in_ptr[0] / num_classes_per_rank;
    const K last_in = in_ptr[num - 1] / num_classes_per_rank;
    if (i <= first_in) {
      out_ptr[i] = 0;
    } else if (i > last_in) {
      out_ptr[i] = num;
    }
  }
}

template<typename T, typename K, int32_t N>
__global__ void PartitionGpu(const int64_t n, const int64_t parallel_num,
                             const int64_t num_classes_per_rank, const K* partion_bound_index,
                             Param<T, K, N> param) {
  const K num = param.in_size[0];
  CUDA_1D_KERNEL_LOOP(i, num) {
#pragma unroll
    for (int32_t j = 0; j < param.num_out; ++j) {
      const int32_t partion_bound_index_start = partion_bound_index[j];
      if (i >= partion_bound_index_start && i < partion_bound_index[j + 1]) {
        const int32_t lower_bound = (param.range_start + j) * num_classes_per_rank;
        param.out[j][i - partion_bound_index_start] = param.in[i] - lower_bound;
        break;
      }
    }
  }
  CUDA_1D_KERNEL_LOOP(i, param.num_out) {
    param.count[i][0] = partion_bound_index[i + 1] - partion_bound_index[i];
  }
}

template<typename T, typename K>
class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(void* ptr, const int64_t n, const int64_t parallel_num) : ptr_(ptr) {
    const size_t unique_in_bytes = GetCudaAlignedSize(n * sizeof(T));
    const size_t in_count_bytes = GetCudaAlignedSize(n * sizeof(K));
    const size_t in_num_unique_bytes = GetCudaAlignedSize(1 * sizeof(K));
    UniqueOpUtil::GetUniqueWithCountsWorkspaceSizeInBytes(
        DeviceType::kGPU, GetDataType<T>::value, GetDataType<K>::value, n, &workspace_bytes_);
    const size_t bound_index_bytes = GetCudaAlignedSize((parallel_num + 1) * sizeof(K));

    unique_in_offset_ = 0;
    in_count_offset_ = unique_in_offset_ + unique_in_bytes;
    in_num_unique_offset_ = in_count_offset_ + in_count_bytes;
    workspace_offset_ = in_num_unique_offset_ + in_num_unique_bytes;
    bound_index_offset_ = workspace_offset_ + workspace_bytes_;

    total_buffer_size_ = unique_in_bytes + in_count_bytes + in_num_unique_bytes + workspace_bytes_
                         + bound_index_bytes;
  }
  ~TmpBufferManager() = default;

  size_t GetTotalBufferSize() const { return total_buffer_size_; }
  int64_t GetWorkspaceBytes() const { return workspace_bytes_; }
  T* UniqueInPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + unique_in_offset_);
  }
  K* InCountPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + in_count_offset_);
  }
  K* InNumUniquePtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + in_num_unique_offset_);
  }
  char* WorkspacePtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<char*>(reinterpret_cast<char*>(ptr_) + workspace_offset_);
  }
  K* BoundIndexPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + bound_index_offset_);
  }

 private:
  size_t unique_in_offset_;
  size_t in_count_offset_;
  size_t in_num_unique_offset_;
  size_t workspace_offset_;
  size_t bound_index_offset_;
  size_t total_buffer_size_;
  int64_t workspace_bytes_;
  void* ptr_;
};

}  // namespace

template<typename T, typename K>
class GatherDispatchKernel final : public user_op::OpKernel {
 public:
  GatherDispatchKernel() = default;
  ~GatherDispatchKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* idx = ctx->Tensor4ArgNameAndIndex("idx", 0);
    const int64_t elem_cnt = indices->shape().elem_cnt();
    const int64_t parallel_num = ctx->Attr<int64_t>("parallel_num");
    const int64_t num_classes = ctx->Attr<int64_t>("num_classes");
    CHECK_EQ(num_classes % parallel_num, 0);
    const int64_t num_classes_per_rank = num_classes / parallel_num;
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    TmpBufferManager<T, K> buffer_manager(tmp_buffer->mut_dptr(), elem_cnt, parallel_num);

    UniqueKernelUtil<DeviceType::kGPU, T, K>::UniqueWithCounts(
        ctx->device_ctx(), elem_cnt, indices->dptr<T>(), buffer_manager.InNumUniquePtr(),
        buffer_manager.UniqueInPtr(), idx->mut_dptr<K>(), buffer_manager.InCountPtr(),
        buffer_manager.WorkspacePtr(), buffer_manager.GetWorkspaceBytes());

    CHECK_EQ(ctx->user_op_conf().output_size("out"), parallel_num);
    CHECK_EQ(ctx->user_op_conf().output_size("count"), parallel_num);
    GetPartionBoundIndex<T, K><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                 ctx->device_ctx()->cuda_stream()>>>(
        elem_cnt, parallel_num, num_classes_per_rank, buffer_manager.UniqueInPtr(),
        buffer_manager.InNumUniquePtr(), buffer_manager.BoundIndexPtr());
    Param<T, K, 128> para;
    para.in = buffer_manager.UniqueInPtr();
    para.in_size = buffer_manager.InNumUniquePtr();
    int64_t remain_size = parallel_num;
    int64_t output_id = 0;
    while (remain_size > 0) {
      para.range_start = output_id;
      int64_t num_out = 0;
      if (remain_size > 128) {
        remain_size -= 128;
        para.num_out = 128;
      } else {
        para.num_out = remain_size;
        remain_size = 0;
      }
      for (int32_t i = 0; i < para.num_out; ++i) {
        user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", output_id);
        user_op::Tensor* count = ctx->Tensor4ArgNameAndIndex("count", output_id);
        output_id++;
        para.out[i] = out->mut_dptr<T>();
        para.count[i] = count->mut_dptr<K>();
      }
      PartitionGpu<T, K, 128><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                ctx->device_ctx()->cuda_stream()>>>(
          elem_cnt, parallel_num, num_classes_per_rank,
          buffer_manager.BoundIndexPtr() + para.range_start, para);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GATHER_DISPATCH_KERNEL(dtype, ktype)                                     \
  REGISTER_USER_KERNEL("gather_dispatch")                                                 \
      .SetCreateFn<GatherDispatchKernel<dtype, ktype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                 \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)    \
                       & (user_op::HobDataType("count", 0) == GetDataType<ktype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                 \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("indices", 0);                 \
        const int64_t parallel_num = ctx->Attr<int64_t>("parallel_num");                  \
        TmpBufferManager<dtype, ktype> buffer_manager(nullptr, in_shape->elem_cnt(),      \
                                                      parallel_num);                      \
        return buffer_manager.GetTotalBufferSize();                                       \
      });

REGISTER_GATHER_DISPATCH_KERNEL(int32_t, int32_t)
REGISTER_GATHER_DISPATCH_KERNEL(int64_t, int32_t)
REGISTER_GATHER_DISPATCH_KERNEL(int32_t, int64_t)
REGISTER_GATHER_DISPATCH_KERNEL(int64_t, int64_t)

}  // namespace oneflow
