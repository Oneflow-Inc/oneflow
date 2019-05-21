#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T>
static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
                 enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k, const T alpha,
                 const T* a, const T* b, const T beta, T* c) {
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;

  cblas_gemm<T>(order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template<typename T>
static void BlobGemmImpl(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                         T alpha, T beta, const Blob* a, const Blob* b, Blob* c) {
  const int m = c->shape().At(0);
  const int n = c->shape().Count(1);
  const int k = (trans_a == CblasNoTrans) ? a->shape().Count(1) : a->shape().At(0);

  NewKernelUtil<kGPU>::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a->dptr<T>(), b->dptr<T>(),
                              beta, c->mut_dptr<T>());
}

template<typename T>
static void ReluImpl(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  T zero = ZeroVal<T>::value;
  for (int64_t i = 0; i != n; ++i) { y[i] = std::max(x[i], zero); }
}

template<typename T>
static void ReluBackwardImpl(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                             T* dx) {
  T zero = ZeroVal<T>::value;
  for (int64_t i = 0; i != n; ++i) { dx[i] = (y[i] > zero) * dy[i]; }
}

template<typename T>
static void AxpyImpl(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                     const int incy) {
  FOR_RANGE(int, i, 0, n) {
    *y += alpha * *x;
    x += incx;
    y += incy;
  }
}

template<typename T>
static void SigmoidImpl(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
  T half = static_cast<T>(0.5);
  for (int64_t i = 0; i != n; ++i) { y[i] = half * std::tanh(half * x[i]) + half; }
}

template<typename T>
static void SigmoidBackwardImpl(DeviceCtx* ctx, const int64_t n, const T* x, const T* y,
                                const T* dy, T* dx) {
  for (int64_t i = 0; i != n; ++i) { dx[i] = y[i] * (1 - y[i]) * dy[i]; }
}

template<typename T>
static void TanHImpl(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
  for (int64_t i = 0; i != n; ++i) { y[i] = std::tanh(x[i]); }
}

template<typename T>
static void TanHBackwardImpl(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                             T* dx) {
  for (int64_t i = 0; i != n; ++i) { dx[i] = (1 - y[i] * y[i]) * dy[i]; }
}

template<typename T>
void BatchedGemmImpl(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                     int batch_size, int m, int n, int k, const T alpha, const T* a, const T* b,
                     const T beta, T* c, T** buf) {
  const int a_stride = m * k;
  const int b_stride = k * n;
  const int c_stride = m * n;
  FOR_RANGE(int32_t, i, 0, batch_size) {
    NewKernelUtil<DeviceType::kCPU>::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a + i * a_stride,
                                            b + i * b_stride, beta, c + i * c_stride);
  }
}

void ComputeOffset(const int32_t num_axes, const int64_t* shape, const int32_t* permutation,
                   std::vector<int64_t>& offset) {
  offset.resize(num_axes);
  std::vector<int64_t> buff(num_axes);
  int64_t cur_offset = 1;
  for (int32_t i = num_axes - 1; i >= 0; --i) {
    buff[i] = cur_offset;
    cur_offset *= shape[i];
  }
  for (int32_t i = 0; i < num_axes; ++i) { offset[permutation[i]] = buff[i]; }
}

void IncreaseIndex(const int64_t* shape, std::vector<int64_t>& index) {
  for (int32_t i = index.size() - 1; i >= 0; --i) {
    ++index[i];
    if (index[i] >= shape[i]) {
      index[i] -= shape[i];
    } else {
      break;
    }
  }
}

template<typename T>
void TransposeImpl(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                   const Shape& y_shape, const PbRf<int32_t>& permutation, const int64_t elem_cnt,
                   const T* x, T* y) {
  int64_t block_size = 1;
  int32_t shared_idxs_num = 0;
  for (int32_t i = num_axis - 1; i >= 0 && permutation[i] == i; --i) {
    block_size *= y_shape.At(i);
    ++shared_idxs_num;
  }
  if (num_axis < 2 || shared_idxs_num == num_axis) {
    memcpy(y, x, elem_cnt * sizeof(T));
    return;
  }
  int32_t trans_axis = num_axis - shared_idxs_num;
  std::vector<int64_t> x_to_y_offset;
  ComputeOffset(trans_axis, y_shape.dim_vec().data(), permutation.data(), x_to_y_offset);
  std::vector<int64_t> x_index_digits(trans_axis, 0);
  int64_t num_blocks = elem_cnt / block_size;
  FOR_RANGE(int64_t, x_idx, 0, num_blocks) {
    int64_t y_idx = std::inner_product(x_to_y_offset.cbegin(), x_to_y_offset.cend(),
                                       x_index_digits.cbegin(), 0);
    if (block_size == 1) {
      y[y_idx] = x[x_idx];
    } else {
      memcpy(y + block_size * y_idx, x + block_size * x_idx, block_size * sizeof(T));
    }
    IncreaseIndex(x_shape.dim_vec().data(), x_index_digits);
  }
}

}  // namespace

#define CPU_KU_METHOD void NewKernelUtil<DeviceType::kCPU>::

CPU_KU_METHOD BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       float alpha, float beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl<float>(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}

CPU_KU_METHOD BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       double alpha, double beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl<double>(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}

CPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const float alpha, const float* a,
                     const float* b, const float beta, float* c) {
  Gemm<float>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

CPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const double alpha, const double* a,
                     const double* b, const double beta, double* c) {
  Gemm<double>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

CPU_KU_METHOD OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                            enum CBLAS_TRANSPOSE trans_b, const int batch_size, const int m,
                            const int n, const int k, const float alpha, const float* a,
                            const float* b, const float beta, float* c, float** buf) {
  BatchedGemmImpl<float>(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k, alpha, a, b,
                         beta, c, buf);
}

CPU_KU_METHOD OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                            enum CBLAS_TRANSPOSE trans_b, const int batch_size, const int m,
                            const int n, const int k, const double alpha, const double* a,
                            const double* b, const double beta, double* c, double** buf) {
  BatchedGemmImpl<double>(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k, alpha, a, b,
                          beta, c, buf);
}

CPU_KU_METHOD Relu(DeviceCtx* ctx, const int64_t n, const float* x, float* y) {
  ReluImpl<float>(ctx, n, x, y);
}

CPU_KU_METHOD Relu(DeviceCtx* ctx, const int64_t n, const double* x, double* y) {
  ReluImpl<double>(ctx, n, x, y);
}

CPU_KU_METHOD ReluBackward(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                           const float* dy, float* dx) {
  ReluBackwardImpl<float>(ctx, n, x, y, dy, dx);
}

CPU_KU_METHOD ReluBackward(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                           const double* dy, double* dx) {
  ReluBackwardImpl<double>(ctx, n, x, y, dy, dx);
}

CPU_KU_METHOD Axpy(DeviceCtx* ctx, const int n, const float alpha, const float* x, const int incx,
                   float* y, const int incy) {
  AxpyImpl<float>(ctx, n, alpha, x, incx, y, incy);
}

CPU_KU_METHOD Axpy(DeviceCtx* ctx, const int n, const double alpha, const double* x, const int incx,
                   double* y, const int incy) {
  AxpyImpl<double>(ctx, n, alpha, x, incx, y, incy);
}

CPU_KU_METHOD Sigmoid(DeviceCtx* ctx, int64_t n, const float* x, float* y) {
  SigmoidImpl<float>(ctx, n, x, y);
}

CPU_KU_METHOD Sigmoid(DeviceCtx* ctx, int64_t n, const double* x, double* y) {
  SigmoidImpl<double>(ctx, n, x, y);
}

CPU_KU_METHOD SigmoidBackward(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                              const float* dy, float* dx) {
  SigmoidBackwardImpl<float>(ctx, n, x, y, dy, dx);
}

CPU_KU_METHOD SigmoidBackward(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                              const double* dy, double* dx) {
  SigmoidBackwardImpl<double>(ctx, n, x, y, dy, dx);
}

CPU_KU_METHOD TanH(DeviceCtx* ctx, const int64_t n, const float* x, float* y) {
  TanHImpl<float>(ctx, n, x, y);
}

CPU_KU_METHOD TanH(DeviceCtx* ctx, const int64_t n, const double* x, double* y) {
  TanHImpl<double>(ctx, n, x, y);
}

CPU_KU_METHOD TanHBackward(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                           const float* dy, float* dx) {
  TanHBackwardImpl<float>(ctx, n, x, y, dy, dx);
}

CPU_KU_METHOD TanHBackward(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                           const double* dy, double* dx) {
  TanHBackwardImpl<double>(ctx, n, x, y, dy, dx);
}

CPU_KU_METHOD Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                        const Shape& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const float* x, float* y) {
  TransposeImpl<float>(ctx, num_axis, x_shape, y_shape, permutation, elem_cnt, x, y);
}

CPU_KU_METHOD Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                        const Shape& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const double* x, double* y) {
  TransposeImpl<double>(ctx, num_axis, x_shape, y_shape, permutation, elem_cnt, x, y);
}

}  // namespace oneflow
