#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/register_manager.h"

namespace oneflow {

namespace {

template<typename T>
void RngUniform(const int64_t elem_cnt, const T min, const T max,
                uint32_t random_seed, T* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_LE(min, max);
  std::mt19937 generator(random_seed);
  std::uniform_real_distribution<T> random_distribution(
      min, std::nextafter(max, std::numeric_limits<T>::max()));

  for (int64_t i = 0; i < elem_cnt; ++i) {
    dptr[i] = random_distribution(generator);
  }
}

template<typename T>
void RngNormal(const int64_t elem_cnt, const T mean, const T std,
               uint32_t random_seed, T* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_GT(std, 0.0);
  std::mt19937 generator(random_seed);
  std::normal_distribution<T> random_distribution(mean, std);

  for (int64_t i = 0; i < elem_cnt; ++i) {
    dptr[i] = random_distribution(generator);
  }
}

template<typename T>
void ConstantInitializer(const ConstantInitializerConf& initializer_conf,
                         Blob* blob) {
  T* dptr = blob->mut_dptr<T>();
  const int64_t elem_cnt = blob->shape().elem_cnt();
  const T value = initializer_conf.value();
  CHECK(elem_cnt);
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = value; }
}

template<typename T>
void RandomUniformInitializer(
    const RandomUniformInitializerConf& initializer_conf, uint32_t random_seed,
    Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  RngUniform<T>(
      blob->shape().elem_cnt(), static_cast<T>(initializer_conf.min()),
      static_cast<T>(initializer_conf.max()), random_seed, blob->mut_dptr<T>());
}
template<typename T>
void RandomNormalInitializer(
    const RandomNormalInitializerConf& initializer_conf, uint32_t random_seed,
    Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  RngNormal<T>(
      blob->shape().elem_cnt(), static_cast<T>(initializer_conf.mean()),
      static_cast<T>(initializer_conf.std()), random_seed, blob->mut_dptr<T>());
}

template<typename T>
T GenInitialFan(VarianceNorm variance_norm, Blob* blob) {
  T fan = ZeroVal<T>::value;
  T fan_in = static_cast<T>(blob->shape().Count(1));
  T fan_out = static_cast<T>(blob->shape().Count(0) / blob->shape().At(1));
  if (variance_norm == VarianceNorm::kAverage) {
    fan = (fan_in + fan_out) / static_cast<T>(2);
  } else if (variance_norm == VarianceNorm::kFanIn) {
    fan = fan_in;
  } else if (variance_norm == VarianceNorm::kFanOut) {
    fan = fan_out;
  } else {
    UNIMPLEMENTED();
  }
  return fan;
}

template<typename T>
void XavierInitializer(const XavierInitializerConf& initializer_conf,
                       uint32_t random_seed, Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  VarianceNorm variance_norm =
      static_cast<VarianceNorm>(initializer_conf.variance_norm());
  T scale =
      std::sqrt(static_cast<T>(3) / GenInitialFan<T>(variance_norm, blob));
  RngUniform<T>(blob->shape().elem_cnt(), static_cast<T>(-scale),
                static_cast<T>(scale), random_seed, blob->mut_dptr<T>());
}

template<typename T>
void MsraInitializer(const MsraInitializerConf& initializer_conf,
                     uint32_t random_seed, Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  VarianceNorm variance_norm =
      static_cast<VarianceNorm>(initializer_conf.variance_norm());
  T std = std::sqrt(static_cast<T>(2) / GenInitialFan<T>(variance_norm, blob));
  RngNormal<T>(blob->shape().elem_cnt(), ZeroVal<T>::value, static_cast<T>(std),
               random_seed, blob->mut_dptr<T>());
}

}  // namespace

template<>
void Memcpy<DeviceType::kCPU>(DeviceCtx* ctx, void* dst, const void* src,
                              size_t sz
#ifdef WITH_CUDA
                              ,
                              cudaMemcpyKind kind
#endif

) {
  memcpy(dst, src, sz);
}

template<>
void Memset<DeviceType::kCPU>(DeviceCtx* ctx, void* dst, const char value,
                              size_t sz) {
  memset(dst, value, sz);
}

template<typename T>
struct KernelUtil<DeviceType::kCPU, T> final {
  static void Dot(DeviceCtx* ctx, const int n, const T* x, const int incx,
                  const T* y, const int incy, T* result) {
    *result = cblas_dot<T>(n, x, incx, y, incy);
  }
  static void Copy(DeviceCtx* ctx, const int n, const T* x, const int incx,
                   T* y, const int incy) {
    cblas_copy<T>(n, x, incx, y, incy);
  }
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x,
                   const int incx, T* y, const int incy) {
    cblas_axpy<T>(n, alpha, x, incx, y, incy);
  }
  static void Axpy(DeviceCtx* ctx, const int n, const T* alpha, const T* x,
                   const int incx, T* y, const int incy) {
    Axpy(ctx, n, *alpha, x, incx, y, incy);
  }
  static void Scal(DeviceCtx* ctx, const int n, const T alpha, T* x,
                   const int incx) {
    cblas_scal<T>(n, alpha, x, incx);
  }
  static void Scal(DeviceCtx* ctx, const int n, const T* alpha, T* x,
                   const int incx) {
    Scal(ctx, n, *alpha, x, incx);
  }
  static void Rsqrt(DeviceCtx* ctx, const int64_t n, T* x,
                    const float epsilon) {
    for (int64_t i = 0; i < n; ++i) { x[i] = 1.0 / std::sqrt(x[i] + epsilon); }
  }
  static void Gemv(DeviceCtx* ctx, const enum CBLAS_TRANSPOSE trans, int m,
                   int n, const T alpha, const T* a, int lda, const T* x,
                   const int incx, const T beta, T* y, const int incy) {
    // Set col major to keep it as the same with cublas
    cblas_gemv<T>(CBLAS_ORDER::CblasColMajor, trans, m, n, alpha, a, lda, x,
                  incx, beta, y, incy);
  }
  static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                   const enum CBLAS_TRANSPOSE trans_a,
                   const enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
                   const int k, const T alpha, const T* a, const int lda,
                   const T* b, const int ldb, const T beta, T* c,
                   const int ldc) {
    cblas_gemm<T>(order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta,
                  c, ldc);
  }

  static void Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr) {
    Max(ctx, n, x, max_ptr, nullptr, 0);
  }
  static void Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr,
                  T* temp_storage, size_t temp_storage_bytes) {
    *max_ptr = *std::max_element(x, x + n);
  }
  static void Exp(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    for (int64_t i = 0; i < n; ++i) { y[i] = std::exp(x[i]); }
  }
  static void Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr) {
    Sum(ctx, n, x, sum_ptr, nullptr, 0);
  }
  static void Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr,
                  T* temp_storage, size_t temp_storage_bytes) {
    *sum_ptr = 0;
    for (int64_t i = 0; i < n; ++i) { *sum_ptr += x[i]; }
  }
  static void Div(DeviceCtx* ctx, const int64_t n, T* x, const T* alpha) {
    for (int64_t i = 0; i < n; ++i) { x[i] = x[i] / (*alpha); }
  }
  static void Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y,
                  T* z) {
    for (int64_t i = 0; i < n; ++i) { z[i] = x[i] * y[i]; }
  }

  static void Sigmoid(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    T half = static_cast<T>(0.5);
    for (int64_t i = 0; i != n; ++i) {
      y[i] = half * std::tanh(half * x[i]) + half;
    }
  }
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x,
                              const T* y, const T* dy, T* dx) {
    for (int64_t i = 0; i != n; ++i) { dx[i] = y[i] * (1 - y[i]) * dy[i]; }
  }
  static void TanH(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    for (int64_t i = 0; i != n; ++i) { y[i] = std::tanh(x[i]); }
  }
  static void TanHBackward(DeviceCtx* ctx, const int64_t n, const T* x,
                           const T* y, const T* dy, T* dx) {
    for (int64_t i = 0; i != n; ++i) { dx[i] = (1 - y[i] * y[i]) * dy[i]; }
  }
  static void Relu(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    T zero = ZeroVal<T>::value;
    for (int64_t i = 0; i != n; ++i) { y[i] = std::max(x[i], zero); }
  }
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x,
                           const T* y, const T* dy, T* dx) {
    T zero = ZeroVal<T>::value;
    for (int64_t i = 0; i != n; ++i) { dx[i] = (y[i] > zero) * dy[i]; }
  }

  static void Initialize(DeviceCtx* ctx,
                         const InitializerConf& initializer_conf,
                         uint32_t random_seed, Blob* blob) {
    if (initializer_conf.has_constant_conf()) {
      ConstantInitializer<T>(initializer_conf.constant_conf(), blob);
    } else if (initializer_conf.has_random_uniform_conf()) {
      RandomUniformInitializer<T>(initializer_conf.random_uniform_conf(),
                                  random_seed, blob);
    } else if (initializer_conf.has_random_normal_conf()) {
      RandomNormalInitializer<T>(initializer_conf.random_normal_conf(),
                                 random_seed, blob);
    } else if (initializer_conf.has_xavier_conf()) {
      XavierInitializer<T>(initializer_conf.xavier_conf(), random_seed, blob);
    } else if (initializer_conf.has_msra_conf()) {
      MsraInitializer<T>(initializer_conf.msra_conf(), random_seed, blob);
    } else {
      UNIMPLEMENTED();
    }
  }
  static void Initialize(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                         const std::string& model_dir, Blob* blob,
                         const std::string& bn_in_op, int32_t dim_num,
                         int64_t num_in_each_dim) {
    int64_t blob_size = blob->ByteSizeOfDataContentField();
    int64_t byte_size_of_each_dim = num_in_each_dim * sizeof(T);
    std::string file_path = JoinPath(model_dir, bn_in_op);
    uint64_t file_size = GlobalFS()->GetFileSize(file_path);
    CHECK_EQ(file_size, dim_num * byte_size_of_each_dim);
    BalancedSplitter splitter = BalancedSplitter(dim_num, part_num);
    int64_t begin_pos = splitter.At(part_id).begin() * byte_size_of_each_dim;
    NormalPersistentInStream in_stream(GlobalFS(), file_path, begin_pos);
    in_stream.Read(blob->mut_dptr<char>(), blob_size);
  }
};

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct KernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

#define DEFINE_INT_KERNEL_UTIL(T, type_proto)                                 \
  template void KernelUtil<DeviceType::kCPU, T>::Sum(                         \
      DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr);               \
  template void KernelUtil<DeviceType::kCPU, T>::Sum(                         \
      DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr,                \
      T* temp_storage, size_t temp_storage_bytes);                            \
  template void KernelUtil<DeviceType::kCPU, T>::Max(                         \
      DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr,                \
      T* temp_storage, size_t temp_storage_bytes);                            \
  template void KernelUtil<DeviceType::kCPU, T>::Relu(                        \
      DeviceCtx* ctx, const int64_t n, const T* x, T* y);                     \
  template void KernelUtil<DeviceType::kCPU, T>::ReluBackward(                \
      DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,   \
      T* dx);                                                                 \
  template<>                                                                  \
  void KernelUtil<DeviceType::kCPU, T>::Axpy(                                 \
      DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, \
      T* y, const int incy) {                                                 \
    FOR_RANGE(int, i, 0, n) {                                                 \
      *y += alpha * *x;                                                       \
      x += incx;                                                              \
      y += incy;                                                              \
    }                                                                         \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_INT_KERNEL_UTIL, INT_DATA_TYPE_SEQ);

#define DEFINE_UNIMPLEMENTED_KERNEL_UTIL(device_type)               \
  template<>                                                        \
  void KernelUtil<device_type, char>::Axpy(                         \
      DeviceCtx* ctx, const int n, const char alpha, const char* x, \
      const int incx, char* y, const int incy) {                    \
    UNIMPLEMENTED();                                                \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_UNIMPLEMENTED_KERNEL_UTIL, DEVICE_TYPE_SEQ);

}  //  namespace oneflow
