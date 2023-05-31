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

namespace {

#define MaxDims 6
#define MAX2(a, b) ((a) > (b)) ? (a) : (b)
#define MAX3(a, b, c) MAX2(MAX2(a, b), c)

using cuda::elementwise::Packed;

#define DEFINE_BINARY_FUNCTOR(OP, expr)                                                        \
  template<typename T>                                                                         \
  struct OP {                                                                                  \
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a expr b; } \
  };                                                                                           \
  template<>                                                                                   \
  struct OP<half> {                                                                            \
    __device__ __forceinline__ half operator()(const half& a, const half& b) const {           \
      return __float2half(__half2float(a) expr __half2float(b));                               \
    }                                                                                          \
  };

DEFINE_BINARY_FUNCTOR(Add, +)
DEFINE_BINARY_FUNCTOR(Sub, -)
DEFINE_BINARY_FUNCTOR(Mul, *)
DEFINE_BINARY_FUNCTOR(Div, /)
#undef DEFINE_BINARY_FUNCTOR

#define DEFINE_BINARY_OP_GRAD_FUNCTOR(OP, dl_expr, dr_expr)                                       \
  template<typename T>                                                                            \
  struct OP##Grad {                                                                               \
    __device__ __forceinline__ void operator()(const T& dout, const T& a, const T& b, T* da,      \
                                               T* db) const {                                     \
      *da = dl_expr dout;                                                                         \
      *db = dr_expr dout;                                                                         \
    }                                                                                             \
  };                                                                                              \
  template<>                                                                                      \
  struct OP##Grad<half> {                                                                         \
    __device__ __forceinline__ void operator()(const half& hdout, const half& ha, const half& hb, \
                                               half* hda, half* hdb) const {                      \
      float dout, a, b;                                                                           \
      dout = __half2float(hdout), a = __half2float(ha), b = __half2float(hb);                     \
      *hda = __float2half(dl_expr dout);                                                          \
      *hdb = __float2half(dr_expr dout);                                                          \
    }                                                                                             \
  };

DEFINE_BINARY_OP_GRAD_FUNCTOR(Add, 1 *, 1 *)
DEFINE_BINARY_OP_GRAD_FUNCTOR(Sub, 1 *, -1 *)
DEFINE_BINARY_OP_GRAD_FUNCTOR(Mul, b*, a*)
DEFINE_BINARY_OP_GRAD_FUNCTOR(Div, 1 / b*, -a / b / b*)
#undef DEFINE_BINARY_OP_GRAD_FUNCTOR

template<int pack_size, typename IndexType, typename BinaryOp, typename R, typename T1, typename T2,
         typename Store, typename Loader1, typename Loader2>
__global__ void noncontiguous_binary_op_kernel(IndexType n_pack, Store y, Loader1 x1, Loader2 x2) {
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
            const Src* src, Dst* dst = nullptr, bool is_contiguous = false)
      : ndims_(ndims), src_(src), dst_(dst), is_contiguous_(is_contiguous) {
    for (int i = 0; i < ndims; i++) {
      strides_[i] = static_cast<IndexType>(strides[i]);
      fast_integer_math_[i] = fast_integer_math[i];
    }
  }

  OF_DEVICE_FUNCTION IndexType index2offset(IndexType index) {
    IndexType offset = 0;
    IndexType div = 0, mod = 0;
#pragma unroll
    for (int dim = ndims_ - 1; dim >= 0; --dim) {
      if (index == 0) break;
      fast_integer_math_[dim].divmod(index, &div, &mod);
      index = div;
      offset += mod * strides_[dim];
    }
    return offset;
  }

  OF_DEVICE_FUNCTION void load(IndexType idx, Packed<Src, pack_size>* pack) {
    IndexType offset;
    if (is_contiguous_)
      offset = idx * pack_size;
    else
      offset = index2offset(idx);
    *pack = *(reinterpret_cast<const Packed<Src, pack_size>*>(src_ + offset));
  }

  OF_DEVICE_FUNCTION void store(IndexType idx, Packed<Dst, pack_size>* pack) {
    IndexType offset;
    if (is_contiguous_)
      offset = idx * pack_size;
    else
      offset = index2offset(idx);
    *(reinterpret_cast<Packed<Dst, pack_size>*>(dst_ + offset)) = *pack;
  }

  int ndims_;
  int pack_dim_;
  bool is_contiguous_;
  const Src* src_;
  Dst* dst_;
  IndexType strides_[MaxDims];
  FastIntegerMath fast_integer_math_[MaxDims];
};

template<int pack_size, typename IndexType, typename BinaryOp, typename R, typename lhs,
         typename rhs, typename Store, typename Load1, typename Load2>
void launch_noncontiguous_binary_op_kernel(cudaStream_t stream, const IndexType n_pack,
                                           Store& store, Load1& load1, Load2& load2) {
  int num_blocks = 1, block_size = cuda::elementwise::kBlockSize;
  cudaError_t err = cuda::elementwise::GetNumBlocks(n_pack, &num_blocks);
  CHECK(err == cudaSuccess);
  noncontiguous_binary_op_kernel<pack_size, IndexType, BinaryOp, R, lhs, rhs>
      <<<num_blocks, block_size, 0, stream>>>(n_pack, store, load1, load2);
}

template<int pack_size, typename IndexType, typename R, typename lhs, typename rhs, typename Store,
         typename Load1, typename Load2>
void dispatchOp(cudaStream_t stream, const std::string& op, const IndexType n_pack, Store& store,
                Load1& load1, Load2& load2) {
  if (op == "add")
    launch_noncontiguous_binary_op_kernel<pack_size, IndexType, Add<R>, R, lhs, rhs>(
        stream, n_pack, store, load1, load2);
  else if (op == "sub")
    launch_noncontiguous_binary_op_kernel<pack_size, IndexType, Sub<R>, R, lhs, rhs>(
        stream, n_pack, store, load1, load2);
  else if (op == "mul")
    launch_noncontiguous_binary_op_kernel<pack_size, IndexType, Mul<R>, R, lhs, rhs>(
        stream, n_pack, store, load1, load2);
  else if (op == "div")
    launch_noncontiguous_binary_op_kernel<pack_size, IndexType, Div<R>, R, lhs, rhs>(
        stream, n_pack, store, load1, load2);
  else
    UNIMPLEMENTED_THEN_THROW();
}

template<int pack_size, typename IndexType, typename R, typename lhs, typename rhs>
void dispatchInplace(cudaStream_t stream, const bool inplace, const std::string& op,
                     const int& ndims, const IndexType n_pack, const int sizes[MaxDims],
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
                       const int& ndims, const int64_t& n_pack, const int sizes[MaxDims],
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
                      const int& ndims, const int64_t n_pack, int pack_size,
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
class NonContiguousBinaryOpKernel final : public user_op::OpKernel {
 public:
  NonContiguousBinaryOpKernel() = default;
  ~NonContiguousBinaryOpKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* x1 = ctx->Tensor4ArgNameAndIndex("lhs", 0);
    const user_op::Tensor* x2 = ctx->Tensor4ArgNameAndIndex("rhs", 0);
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
      if (x1->stride()[i] == 1 && x2->stride()[i] == 1 && y->stride()[i] == 1) {
        pack_size = 16 / max_elem_size;
        while (pack_size > 1 && sizes[i] % pack_size) pack_size >>= 1;
        sizes[i] = sizes[i] / pack_size;
        strides[0][i] *= pack_size;
        strides[1][i] *= pack_size;
        strides[2][i] *= pack_size;
      }
    }

    dispatchPacksize(ctx->stream()->As<ep::CudaStream>()->cuda_stream(), inplace, op, ndims,
                     elem_cnt / pack_size, pack_size, sizes, strides, y->mut_dptr<R>(),
                     x1->dptr<lhs>(), x2->dptr<rhs>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_USER_KERNEL_NONCONTIGUOUS_BINARY_OP_KERNEL(dtype, lhs, rhs)          \
  REGISTER_USER_KERNEL("noncontiguous_binary_op")                                     \
      .SetCreateFn<NonContiguousBinaryOpKernel<dtype, lhs, rhs>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                \
                       && (user_op::HobDataType("y", 0) == GetDataType<lhs>::value)   \
                       && (user_op::HobDataType("lhs", 0) == GetDataType<lhs>::value) \
                       && (user_op::HobDataType("rhs", 0) == GetDataType<rhs>::value));

// output_type, lhs_type, rhs_type
REGISTER_USER_KERNEL_NONCONTIGUOUS_BINARY_OP_KERNEL(float, float, float)
REGISTER_USER_KERNEL_NONCONTIGUOUS_BINARY_OP_KERNEL(half, half, half)
// #if CUDA_VERSION >= 11000
// REGISTER_USER_KERNEL_NONCONTIGUOUS_BINARY_OP_KERNEL(nv_bfloat16, nv_bfloat16, nv_bfloat16)
// #endif

// ------------------------------------- grad kernel -------------------------------------
template<int pack_size, typename IndexType, typename BinaryOp, typename R, typename T1, typename T2,
         typename Loadery, typename Loader1, typename Loader2>
__global__ void noncontiguous_binary_op_grad_kernel(IndexType n_pack, Loadery dy, Loader1 load1,
                                                    Loader2 load2) {
  Packed<R, pack_size> pack_dy;
  Packed<T1, pack_size> pack_x1;
  Packed<T2, pack_size> pack_x2;
  Packed<T1, pack_size> pack_dx1;
  Packed<T2, pack_size> pack_dx2;
  CUDA_1D_KERNEL_LOOP_T(IndexType, i, n_pack) {
    load1.load(i, &pack_x1);
    load2.load(i, &pack_x2);
    dy.load(i, &pack_dy);
#pragma unroll
    for (int j = 0; j < pack_size; ++j)
      BinaryOp()(pack_dy.elem[j], pack_x1.elem[j], pack_x2.elem[j], &pack_dx1.elem[j],
                 &pack_dx2.elem[j]);  // todo: Apply2
    load1.store(i, &pack_dx1);
    load2.store(i, &pack_dx2);
  }
};

template<int pack_size, typename IndexType, typename BinaryOp, typename R, typename lhs,
         typename rhs, typename Loady, typename Load1, typename Load2>
void launch_noncontiguous_binary_op_grad_kernel(cudaStream_t stream, const IndexType n_pack,
                                                Loady& load_y, Load1& load1, Load2& load2) {
  int num_blocks = 1, block_size = cuda::elementwise::kBlockSize;
  cudaError_t err = cuda::elementwise::GetNumBlocks(n_pack, &num_blocks);
  CHECK(err == cudaSuccess);
  noncontiguous_binary_op_grad_kernel<pack_size, IndexType, BinaryOp, R, lhs, rhs>
      <<<num_blocks, block_size, 0, stream>>>(n_pack, load_y, load1, load2);
}

template<int pack_size, typename IndexType, typename R, typename lhs, typename rhs, typename Loady,
         typename Load1, typename Load2>
void dispatchOpGrad(cudaStream_t stream, const std::string& op, const IndexType& n_pack,
                    Loady& load_y, Load1& load1, Load2& load2) {
  if (op == "add")
    launch_noncontiguous_binary_op_grad_kernel<pack_size, IndexType, AddGrad<R>, R, lhs, rhs>(
        stream, n_pack, load_y, load1, load2);
  else if (op == "sub")
    launch_noncontiguous_binary_op_grad_kernel<pack_size, IndexType, SubGrad<R>, R, lhs, rhs>(
        stream, n_pack, load_y, load1, load2);
  else if (op == "mul")
    launch_noncontiguous_binary_op_grad_kernel<pack_size, IndexType, MulGrad<R>, R, lhs, rhs>(
        stream, n_pack, load_y, load1, load2);
  else if (op == "div")
    launch_noncontiguous_binary_op_grad_kernel<pack_size, IndexType, DivGrad<R>, R, lhs, rhs>(
        stream, n_pack, load_y, load1, load2);
  else
    UNIMPLEMENTED_THEN_THROW();
}

template<int pack_size, typename IndexType, typename R, typename lhs, typename rhs>
void dispatchLoader(cudaStream_t stream, const std::string& op, const int& ndims,
                    const IndexType n_pack, const int sizes[MaxDims], const int strides[][MaxDims],
                    lhs* dx1, rhs* dx2, const R* dy, const lhs* x1, const rhs* x2) {
  typedef FastIntegerMath<IndexType> FastIntegerMathT;
  FastIntegerMathT fast_integer_math[MaxDims];
  for (int i = 0; i < ndims; ++i) fast_integer_math[i] = FastIntegerMathT(sizes[i]);
  LoadStore<pack_size, IndexType, FastIntegerMathT, lhs, R> load_y(fast_integer_math, ndims,
                                                                   strides[0], dy);
  LoadStore<pack_size, IndexType, FastIntegerMathT, lhs, lhs> loader_store1(
      fast_integer_math, ndims, strides[1], x1, dx1);

  LoadStore<pack_size, IndexType, FastIntegerMathT, rhs, rhs> loader_store2(
      fast_integer_math, ndims, strides[2], x2, dx2);
  dispatchOpGrad<pack_size, IndexType, R, lhs, rhs>(stream, op, n_pack, load_y, loader_store1,
                                                    loader_store2);
}

template<int pack_size, typename R, typename lhs, typename rhs>
void dispatchIndexTypeGrad(cudaStream_t stream, const std::string& op, const int& ndims,
                           const int64_t& n_pack, const int sizes[MaxDims],
                           const int strides[][MaxDims], lhs* dx1, rhs* dx2, const R* dy,
                           const lhs* x1, const rhs* x2) {
  if ((n_pack * pack_size) >> 30 == 0) {
    int32_t n = (int32_t)n_pack;
    dispatchLoader<pack_size, int32_t, R, lhs, rhs>(stream, op, ndims, n, sizes, strides, dx1, dx2,
                                                    dy, x1, x2);
  } else
    dispatchLoader<pack_size, int64_t, R, lhs, rhs>(stream, op, ndims, n_pack, sizes, strides, dx1,
                                                    dx2, dy, x1, x2);
}

template<typename R, typename lhs, typename rhs>
void dispatchPacksizeGrad(cudaStream_t stream, const std::string& op, const int& ndims,
                          const int64_t& n_pack, int& pack_size, const int sizes[MaxDims],
                          const int strides[][MaxDims], lhs* dx1, rhs* dx2, const R* dy,
                          const lhs* x1, const rhs* x2) {
  if (pack_size == 8)
    dispatchIndexTypeGrad<8, R, lhs, rhs>(stream, op, ndims, n_pack, sizes, strides, dx1, dx2, dy,
                                          x1, x2);
  else if (pack_size == 4)
    dispatchIndexTypeGrad<4, R, lhs, rhs>(stream, op, ndims, n_pack, sizes, strides, dx1, dx2, dy,
                                          x1, x2);
  else if (pack_size == 2)
    dispatchIndexTypeGrad<2, R, lhs, rhs>(stream, op, ndims, n_pack, sizes, strides, dx1, dx2, dy,
                                          x1, x2);
  else if (pack_size == 1)
    dispatchIndexTypeGrad<1, R, lhs, rhs>(stream, op, ndims, n_pack, sizes, strides, dx1, dx2, dy,
                                          x1, x2);
  else
    UNIMPLEMENTED();
}

template<typename R, typename lhs, typename rhs>
class NonContiguousBinaryOpGradKernel final : public user_op::OpKernel {
 public:
  NonContiguousBinaryOpGradKernel() = default;
  ~NonContiguousBinaryOpGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x1 = ctx->Tensor4ArgNameAndIndex("lhs", 0);
    const user_op::Tensor* x2 = ctx->Tensor4ArgNameAndIndex("rhs", 0);
    user_op::Tensor* dx1 = ctx->Tensor4ArgNameAndIndex("dlhs", 0);
    user_op::Tensor* dx2 = ctx->Tensor4ArgNameAndIndex("drhs", 0);
    const std::string op = ctx->Attr<std::string>("op");
    const bool inplace = ctx->Attr<bool>("inplace");
    CHECK(inplace == false) << "inplace should be set to `false` to compute gradients.";
    int ndims = dy->shape_view().NumAxes();
    const ShapeView& shape = dy->shape_view();
    int sizes[MaxDims];
    int strides[3][MaxDims];

    int pack_size = 1;
    int64_t elem_cnt = 1;
    int max_elem_size = MAX3(GetSizeOfDataType(dy->data_type()), GetSizeOfDataType(x1->data_type()),
                             GetSizeOfDataType(x2->data_type()));
    for (int i = 0; i < ndims; ++i) {
      sizes[i] = shape.At(i);
      elem_cnt *= shape.At(i);
      strides[0][i] = dy->stride()[i];
      strides[1][i] = x1->stride()[i];
      strides[2][i] = x2->stride()[i];
      if (x1->stride()[i] == 1 && x2->stride()[i] == 1 && dy->stride()[i] == 1) {
        pack_size = 16 / max_elem_size;
        while (pack_size > 1 && sizes[i] % pack_size) pack_size >>= 1;
        sizes[i] = sizes[i] / pack_size;
        strides[0][i] *= pack_size;
        strides[1][i] *= pack_size;
        strides[2][i] *= pack_size;
      }
    }

    dispatchPacksizeGrad(ctx->stream()->As<ep::CudaStream>()->cuda_stream(), op, ndims,
                         elem_cnt / pack_size, pack_size, sizes, strides, dx1->mut_dptr<lhs>(),
                         dx2->mut_dptr<rhs>(), dy->dptr<R>(), x1->dptr<lhs>(), x2->dptr<rhs>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_USER_KERNEL_NONCONTIGUOUS_BINARY_OP_GRAD_KERNEL(dtype, lhs, rhs)      \
  REGISTER_USER_KERNEL("noncontiguous_binary_op_grad")                                 \
      .SetCreateFn<NonContiguousBinaryOpGradKernel<dtype, lhs, rhs>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("lhs", 0) == GetDataType<lhs>::value)  \
                       && (user_op::HobDataType("rhs", 0) == GetDataType<rhs>::value));

// output_type, lhs_type, rhs_type
REGISTER_USER_KERNEL_NONCONTIGUOUS_BINARY_OP_GRAD_KERNEL(float, float, float)
REGISTER_USER_KERNEL_NONCONTIGUOUS_BINARY_OP_GRAD_KERNEL(half, half, half)

}  // namespace oneflow
