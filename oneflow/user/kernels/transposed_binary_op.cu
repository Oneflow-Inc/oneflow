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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/include/primitive/fast_integer_math.h"
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

#define MaxDims 6

namespace {
#define MAX2(a, b) ((a) > (b)) ? (a) : (b)
#define MAX3(a, b, c) MAX2(MAX2(a, b), c)

struct Add {
  template<typename R>
  OF_DEVICE_FUNCTION R operator()(R x1, R x2) {
    return x1 + x2;
  }
};
struct Sub {
  template<typename R>
  OF_DEVICE_FUNCTION R operator()(R x1, R x2) {
    return x1 - x2;
  }
};
struct Mul {
  template<typename R>
  OF_DEVICE_FUNCTION R operator()(R x1, R x2) {
    return x1 * x2;
  }
};
struct Div {
  template<typename R>
  OF_DEVICE_FUNCTION R operator()(R x1, R x2) {
    return x1 / x2;
  }
};

template<typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed {
  __device__ Packed() {  // do nothing
  }
  union {
    T elem[pack_size];
  };
};

template<int pack_size, typename IndexType, typename BinaryOp, typename R, typename T1, typename T2,
         typename Store, typename Loader1, typename Loader2>
__global__ void launch_kernel(IndexType n_pack, Store& y, Loader1& x1, Loader2& x2) {
  Packed<R, pack_size> pack_y;
  Packed<T1, pack_size> pack_x1;
  Packed<T2, pack_size> pack_x2;
  CUDA_1D_KERNEL_LOOP_T(IndexType, i, n_pack) {
    x1.load(i, &pack_x1);
    x2.load(i, &pack_x2);
#pragma unroll
    for (int j = 0; j < pack_size; ++j)
      pack_y.elem[j] = BinaryOp()(static_cast<R>(pack_x1.elem[j]),
                                  static_cast<R>(pack_x2.elem[j]));  // todo: Apply2
    y.store(i, &pack_y);
  }
};

template<int pack_size, typename IndexType, typename FastIntegerMath, typename Src,
         typename Dst = void>
struct LoadStore {
  LoadStore(FastIntegerMath fast_integer_math[MaxDims], const int ndims, const int strides[MaxDims],
            const Src* src, Dst* dst = nullptr)
      : ndims_(ndims), src_(src), dst_(dst) {
    for (int i = 0; i < ndims; i++) {
      strides_[i] = static_cast<IndexType>(strides[i]);
      fast_integer_math_[i] = fast_integer_math[i];
    }
    last_idx_ = 0;
    last_offset_ = 0;
  }

  OF_DEVICE_FUNCTION IndexType index2offset(IndexType index) {
    IndexType offset = 0;
    IndexType div = 0, mod = 0;
#pragma unroll
    for (int dim = ndims_ - 1; dim >= 0 && index > 0; --dim) {
      fast_integer_math_[dim].divmod(index, &div, &mod);
      index = div;
      offset += mod * strides_[dim];
    }
    return offset;
  }

  OF_DEVICE_FUNCTION void load(IndexType idx, Packed<Src, pack_size>* pack) {
    IndexType offset = index2offset(idx);
    last_idx_ = idx;
    last_offset_ = offset;
    *pack = *(reinterpret_cast<const Packed<Src, pack_size>*>(src_) + offset);
  }

  OF_DEVICE_FUNCTION void store(IndexType idx, Packed<Dst, pack_size>* pack) {
    IndexType offset = (idx == last_idx_) ? last_offset_ : index2offset(idx);  // inplace
    *(reinterpret_cast<Packed<Dst, pack_size>*>(dst_) + offset) = *pack;
  }

  int ndims_;
  IndexType last_idx_;
  IndexType last_offset_;
  const Src* src_;
  Dst* dst_;
  IndexType strides_[MaxDims];
  FastIntegerMath fast_integer_math_[MaxDims];
};

template<int pack_size, typename IndexType, typename BinaryOp, typename R, typename lhs,
         typename rhs, typename Store, typename Load1, typename Load2>
void launch(cudaStream_t stream, const IndexType n_pack, Store& store, Load1& load1, Load2& load2) {
  int num_blocks = 1, block_size = cuda::elementwise::kBlockSize;
  cudaError_t err = cuda::elementwise::GetNumBlocks(n_pack, &num_blocks);
  CHECK(err == cudaSuccess);
  launch_kernel<pack_size, IndexType, BinaryOp, R, lhs, rhs>
      <<<num_blocks, block_size, 0, stream>>>(n_pack, store, load1, load2);
}

template<int pack_size, typename IndexType, typename R, typename lhs, typename rhs, typename Store,
         typename Load1, typename Load2>
void dispatchOp(cudaStream_t stream, const std::string& op, const IndexType n_pack, Store& store,
                Load1& load1, Load2& load2) {
  if (op == "add")
    launch<pack_size, IndexType, Add, R, lhs, rhs>(stream, n_pack, store, load1, load2);
  else if (op == "sub")
    launch<pack_size, IndexType, Sub, R, lhs, rhs>(stream, n_pack, store, load1, load2);
  else if (op == "mul")
    launch<pack_size, IndexType, Mul, R, lhs, rhs>(stream, n_pack, store, load1, load2);
  else if (op == "div")
    launch<pack_size, IndexType, Div, R, lhs, rhs>(stream, n_pack, store, load1, load2);
  else
    UNIMPLEMENTED_THEN_THROW();
}

template<int pack_size, typename IndexType, typename R, typename lhs, typename rhs>
void dispatchInplace(cudaStream_t stream, const bool inplace, const std::string& op,
                     const int ndims, const IndexType n_pack, const int sizes[MaxDims],
                     const int strides[][MaxDims], R* y, const lhs* x1, const rhs* x2) {
  typedef FastIntegerMath<IndexType> FastIntegerMathT;
  FastIntegerMathT fast_integer_math[MaxDims];
  for (int i = 0; i < ndims; ++i) fast_integer_math[i] = FastIntegerMathT(sizes[i]);
  if (inplace) {
    LoadStore<pack_size, IndexType, FastIntegerMathT, lhs, R> load_store(fast_integer_math, ndims,
                                                                         strides[0], x1, y);
    LoadStore<pack_size, IndexType, FastIntegerMathT, rhs> loader2(fast_integer_math, ndims,
                                                                   strides[2], x2);
    dispatchOp<pack_size, IndexType, R, lhs, rhs>(stream, op, n_pack, load_store, load_store,
                                                  loader2);
  } else {
    LoadStore<pack_size, IndexType, FastIntegerMathT, lhs, R> store(fast_integer_math, ndims,
                                                                    strides[0], nullptr, y);
    LoadStore<pack_size, IndexType, FastIntegerMathT, lhs> loader1(fast_integer_math, ndims,
                                                                   strides[1], x1);

    LoadStore<pack_size, IndexType, FastIntegerMathT, rhs> loader2(fast_integer_math, ndims,
                                                                   strides[2], x2);
    dispatchOp<pack_size, IndexType, R, lhs, rhs>(stream, op, n_pack, store, loader1, loader2);
  }
}

template<int pack_size, typename R, typename lhs, typename rhs>
void dispatchIndexType(cudaStream_t stream, const bool inplace, const std::string& op,
                       const int ndims, const int64_t n_pack, const int sizes[MaxDims],
                       const int strides[][MaxDims], R* y, const lhs* x1, const rhs* x2) {
  if ((n_pack * pack_size) >> 30 == 0) {
    int32_t n = (int32_t)n_pack;
    dispatchInplace<pack_size, int32_t, R, lhs, rhs>(stream, inplace, op, ndims, n, sizes, strides,
                                                     y, x1, x2);
  } else
    dispatchInplace<pack_size, int64_t, R, lhs, rhs>(stream, inplace, op, ndims, n_pack, sizes,
                                                     strides, y, x1, x2);
}

template<typename R, typename lhs, typename rhs>
void dispatchPacksize(cudaStream_t stream, const bool inplace, const std::string& op,
                      const int ndims, const int64_t n_pack, int pack_size,
                      const int sizes[MaxDims], const int strides[][MaxDims], R* y, const lhs* x1,
                      const rhs* x2) {
  if (pack_size == 8)
    dispatchIndexType<8, R, lhs, rhs>(stream, inplace, op, ndims, n_pack, sizes, strides, y, x1,
                                      x2);
  else if (pack_size == 4)
    dispatchIndexType<4, R, lhs, rhs>(stream, inplace, op, ndims, n_pack, sizes, strides, y, x1,
                                      x2);
  else if (pack_size == 2)
    dispatchIndexType<2, R, lhs, rhs>(stream, inplace, op, ndims, n_pack, sizes, strides, y, x1,
                                      x2);
  else if (pack_size == 1)
    dispatchIndexType<1, R, lhs, rhs>(stream, inplace, op, ndims, n_pack, sizes, strides, y, x1,
                                      x2);
  else
    UNIMPLEMENTED();
}
}  // namespace

template<typename R, typename lhs, typename rhs>
class TransposedBinaryOpKernel final : public user_op::OpKernel {
 public:
  TransposedBinaryOpKernel() = default;
  ~TransposedBinaryOpKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* x1 = ctx->Tensor4ArgNameAndIndex("lhs", 0);
    user_op::Tensor* x2 = ctx->Tensor4ArgNameAndIndex("rhs", 0);
    const std::string op = ctx->Attr<std::string>("op");
    const bool inplace = ctx->Attr<bool>("inplace");
    int ndims = y->shape_view().NumAxes();
    const ShapeView& shape = y->shape_view();
    int sizes[MaxDims];
    int strides[3][MaxDims];

    int pack_size = 1;
    int64_t elem_cnt = 1;
    int max_elem_size = MAX3(GetSizeOfDataType(y->data_type()), GetSizeOfDataType(x1->data_type()),
                             GetSizeOfDataType(x2->data_type()));
    for (int i = 0; i < ndims; ++i) {
      sizes[i] = shape.At(i);
      elem_cnt *= shape.At(i);
      strides[0][i] = y->stride()[i];
      strides[1][i] = x1->stride()[i];
      strides[2][i] = x2->stride()[i];
      if (x1->stride()[i] == 1 && x2->stride()[i] == 1) {
        pack_size = 16 / max_elem_size;
        while (pack_size > 1 && sizes[i] % pack_size) pack_size >>= 1;
        sizes[i] = shape.At(i) / pack_size;  //
      }
    }

    dispatchPacksize(ctx->stream()->As<ep::CudaStream>()->cuda_stream(), inplace, op, ndims,
                     elem_cnt / pack_size, pack_size, sizes, strides, y->mut_dptr<R>(),
                     x1->dptr<lhs>(), x2->dptr<rhs>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_USER_KERNEL_TRANSPOSED_BINARY_OP_KERNEL(dtype, lhs, rhs)             \
  REGISTER_USER_KERNEL("transposed_binary_op")                                        \
      .SetCreateFn<TransposedBinaryOpKernel<dtype, lhs, rhs>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                \
                       && (user_op::HobDataType("y", 0) == GetDataType<lhs>::value)   \
                       && (user_op::HobDataType("lhs", 0) == GetDataType<lhs>::value) \
                       && (user_op::HobDataType("rhs", 0) == GetDataType<rhs>::value));

// output_type, lhs_type, rhs_type
REGISTER_USER_KERNEL_TRANSPOSED_BINARY_OP_KERNEL(float, float, float)
REGISTER_USER_KERNEL_TRANSPOSED_BINARY_OP_KERNEL(half, half, half)
#if CUDA_VERSION >= 11000
REGISTER_USER_KERNEL_TRANSPOSED_BINARY_OP_KERNEL(nv_bfloat16, nv_bfloat16, nv_bfloat16)
#endif
}  // namespace oneflow
