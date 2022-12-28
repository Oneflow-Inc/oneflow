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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

constexpr int kBlockSize = sizeof(int64_t) * 8;

template<typename T>
__host__ __device__ __forceinline__ T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template<typename T>
__host__ __device__ __forceinline__ T IoU(T const* const a, T const* const b) {
  T interS =
      max(min(a[2], b[2]) - max(a[0], b[0]), 0.f) * max(min(a[3], b[3]) - max(a[1], b[1]), 0.f);
  T Sa = (a[2] - a[0]) * (a[3] - a[1]);
  T Sb = (b[2] - b[0]) * (b[3] - b[1]);
  return interS / (Sa + Sb - interS);
}

template<typename T>
__global__ void CalcSuppressionBitmaskMatrix(int num_boxes, float iou_threshold, const T* boxes,
                                             int64_t* suppression_bmask_matrix) {
  const int row = blockIdx.y;
  const int col = blockIdx.x;

  if (row > col) return;

  const int row_size = min(num_boxes - row * kBlockSize, kBlockSize);
  const int col_size = min(num_boxes - col * kBlockSize, kBlockSize);

  __shared__ T block_boxes[kBlockSize * 4];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 4 + 0] = boxes[(kBlockSize * col + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] = boxes[(kBlockSize * col + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] = boxes[(kBlockSize * col + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] = boxes[(kBlockSize * col + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = kBlockSize * row + threadIdx.x;
    const T* cur_box_ptr = boxes + cur_box_idx * 4;
    unsigned long long bits = 0;
    int start = 0;
    if (row == col) { start = threadIdx.x + 1; }
    for (int i = start; i < col_size; i++) {
      if (IoU(cur_box_ptr, block_boxes + i * 4) > iou_threshold) { bits |= 1Ull << i; }
    }
    suppression_bmask_matrix[cur_box_idx * gridDim.y + col] = bits;
  }
}

__global__ void ScanSuppression(int num_boxes, int num_blocks, int num_keep,
                                int64_t* suppression_bmask, int8_t* keep_mask) {
  extern __shared__ int64_t remv[];
  remv[threadIdx.x] = 0;
  for (int i = 0; i < num_boxes; ++i) {
    int block_n = i / kBlockSize;
    int block_i = i % kBlockSize;
    if (!(remv[block_n] & (1Ull << block_i))) {
      remv[threadIdx.x] |= suppression_bmask[i * num_blocks + threadIdx.x];
      if (threadIdx.x == block_n && num_keep > 0) {
        keep_mask[i] = 1;
        num_keep -= 1;
      }
    }
  }
}

}  // namespace

template<typename T>
class NmsGpuKernel final : public user_op::OpKernel {
 public:
  NmsGpuKernel() = default;
  ~NmsGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* boxes_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* keep_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_blob = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const T* boxes = boxes_blob->dptr<T>();
    int8_t* keep = keep_blob->mut_dptr<int8_t>();
    int64_t* suppression_mask = tmp_blob->mut_dptr<int64_t>();

    const int num_boxes = boxes_blob->shape_view().At(0);
    int num_keep = ctx->Attr<int>("keep_n");
    if (num_keep <= 0 || num_keep > num_boxes) { num_keep = num_boxes; }
    const int num_blocks = CeilDiv<int>(num_boxes, kBlockSize);
    Memset<DeviceType::kCUDA>(ctx->stream(), suppression_mask, 0,
                              num_boxes * num_blocks * sizeof(int64_t));
    Memset<DeviceType::kCUDA>(ctx->stream(), keep, 0, num_boxes * sizeof(int8_t));

    dim3 blocks(num_blocks, num_blocks);
    dim3 threads(kBlockSize);
    CalcSuppressionBitmaskMatrix<<<blocks, threads, 0,
                                   ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        num_boxes, ctx->Attr<float>("iou_threshold"), boxes, suppression_mask);
    ScanSuppression<<<1, num_blocks, num_blocks * sizeof(int64_t),
                      ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        num_boxes, num_blocks, num_keep, suppression_mask, keep);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_NMS_CUDA_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("nms")                                                           \
      .SetCreateFn<NmsGpuKernel<dtype>>()                                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("out", 0) == DataType::kInt8)           \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                               \
        const Shape& in_shape = ctx->Shape4ArgNameAndIndex("in", 0);                    \
        int64_t num_boxes = in_shape.At(0);                                             \
        int64_t blocks = CeilDiv<int64_t>(num_boxes, kBlockSize);                       \
        return num_boxes * blocks * sizeof(int64_t);                                    \
      });

REGISTER_NMS_CUDA_KERNEL(float)
REGISTER_NMS_CUDA_KERNEL(double)

}  // namespace oneflow
