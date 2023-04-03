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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

template<typename T, typename IDX>
__global__ void BinaryConcatKernel(const IDX out_elems, const IDX out_cols, const IDX in0_cols,
                                   const IDX in1_cols, const T* src0, const T* src1, T* dst) {
  CUDA_1D_KERNEL_LOOP_T(IDX, i, out_elems) {
    const IDX row = i / out_cols;
    const IDX col = i - row * out_cols;
    const T* src_ptr = nullptr;
    if (col < in0_cols) {
      src_ptr = src0 + row * in0_cols + col;
    } else {
      src_ptr = src1 + row * in1_cols + (col - in0_cols);
    }
    dst[i] = *src_ptr;
  }
}

template<typename T, typename IDX>
void LaunchBinaryConcatKernel(ep::Stream* stream, const IDX rows, const IDX in0_cols,
                              const IDX in1_cols, const void* src0, const void* src1, void* dst) {
  const IDX out_cols = in0_cols + in1_cols;
  const IDX out_elems = rows * out_cols;
  RUN_CUDA_KERNEL((BinaryConcatKernel<T, IDX>), stream, out_elems, out_elems, out_cols, in0_cols,
                  in1_cols, reinterpret_cast<const T*>(src0), reinterpret_cast<const T*>(src1),
                  reinterpret_cast<T*>(dst));
}

template<typename T>
void DispatchIndexType(ep::Stream* stream, const int64_t rows, const int64_t in0_cols,
                       const int64_t in1_cols, const void* src0, const void* src1, void* dst) {
  if (rows * (in0_cols + in1_cols) >= (1 >> 30)) {
    LaunchBinaryConcatKernel<T, int64_t>(stream, rows, in0_cols, in1_cols, src0, src1, dst);
  } else {
    LaunchBinaryConcatKernel<T, int32_t>(stream, rows, in0_cols, in1_cols, src0, src1, dst);
  }
}

void DispatchDataType(ep::Stream* stream, const int64_t rows, const int64_t in0_cols,
                      const int64_t in1_cols, const void* src0, const void* src1, void* dst) {
  const uintptr_t src0_ptr = reinterpret_cast<uintptr_t>(src0);
  const uintptr_t src1_ptr = reinterpret_cast<uintptr_t>(src1);
  const uintptr_t dst_ptr = reinterpret_cast<uintptr_t>(dst);
  const auto IsAligned = [&](const size_t alignment) {
    return src0_ptr % alignment == 0 && src1_ptr % alignment == 0 && dst_ptr % alignment == 0
           && in0_cols % alignment == 0 && in1_cols % alignment == 0;
  };
  if (IsAligned(16)) {
    DispatchIndexType<uint4>(stream, rows, in0_cols / 16, in1_cols / 16, src0, src1, dst);
  } else if (IsAligned(8)) {
    DispatchIndexType<uint2>(stream, rows, in0_cols / 8, in1_cols / 8, src0, src1, dst);
  } else if (IsAligned(4)) {
    DispatchIndexType<uint32_t>(stream, rows, in0_cols / 4, in1_cols / 4, src0, src1, dst);
  } else if (IsAligned(2)) {
    DispatchIndexType<uint16_t>(stream, rows, in0_cols / 2, in1_cols / 2, src0, src1, dst);
  } else {
    DispatchIndexType<uint8_t>(stream, rows, in0_cols, in1_cols, src0, src1, dst);
  }
}

void DispatchBinaryConcat(ep::Stream* stream, const int64_t elem_size, const int64_t rows,
                          const int64_t in0_cols, const int64_t in1_cols, const void* src0,
                          const void* src1, void* dst) {
  DispatchDataType(stream, rows, in0_cols * elem_size, in1_cols * elem_size, src0, src1, dst);
}

class ConcatKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  ConcatKernel() = default;
  ~ConcatKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const DataType data_type = out_tensor->data_type();
    if (out_tensor->shape_view().elem_cnt() == 0) { return; }
    const int64_t axis = ctx->Attr<int64_t>("axis");
    CHECK_GE(axis, 0);
    const int64_t num_axes = out_tensor->shape_view().NumAxes();
    CHECK_LT(axis, num_axes);
    const int64_t out_cols = out_tensor->shape_view().Count(axis);
    const int64_t rows = out_tensor->shape_view().elem_cnt() / out_cols;
    CHECK_GT(rows, 0);

    CHECK_EQ(ctx->input_size("in"), 2);
    const user_op::Tensor* in0_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* in1_tensor = ctx->Tensor4ArgNameAndIndex("in", 1);
    CHECK_EQ(in0_tensor->data_type(), data_type);
    CHECK_EQ(in1_tensor->data_type(), data_type);
    if (in0_tensor->shape_view().elem_cnt() == 0) {
      CHECK_EQ(in1_tensor->shape_view(), out_tensor->shape_view());
      Memcpy<DeviceType::kCUDA>(ctx->stream(), out_tensor->mut_dptr(), in1_tensor->dptr(),
                                out_tensor->shape_view().elem_cnt() * GetSizeOfDataType(data_type));
      return;
    }
    if (in1_tensor->shape_view().elem_cnt() == 0) {
      CHECK_EQ(in0_tensor->shape_view(), out_tensor->shape_view());
      Memcpy<DeviceType::kCUDA>(ctx->stream(), out_tensor->mut_dptr(), in0_tensor->dptr(),
                                out_tensor->shape_view().elem_cnt() * GetSizeOfDataType(data_type));
      return;
    }
    CHECK_EQ(in0_tensor->shape_view().NumAxes(), num_axes);
    CHECK_EQ(in1_tensor->shape_view().NumAxes(), num_axes);
    for (int64_t i = 0; i < num_axes; ++i) {
      if (i != axis) {
        CHECK_EQ(in0_tensor->shape_view().At(i), out_tensor->shape_view().At(i));
        CHECK_EQ(in1_tensor->shape_view().At(i), out_tensor->shape_view().At(i));
      }
    }
    CHECK_EQ(in0_tensor->shape_view().At(axis) + in1_tensor->shape_view().At(axis),
             out_tensor->shape_view().At(axis));
    const int64_t in0_cols = in0_tensor->shape_view().Count(axis);
    const int64_t in1_cols = in1_tensor->shape_view().Count(axis);

    DispatchBinaryConcat(ctx->stream(), GetSizeOfDataType(data_type), rows, in0_cols, in1_cols,
                         in0_tensor->dptr(), in1_tensor->dptr(), out_tensor->mut_dptr());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

REGISTER_USER_KERNEL("concat")
    .SetCreateFn<ConcatKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobInputSize("in") == 2))
    .SetPriority(user_op::kKernelPriorityOptimized);

}  // namespace oneflow
