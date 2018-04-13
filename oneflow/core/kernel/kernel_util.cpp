#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/kernel/kernel.h"

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
void RngIntUniform(const int64_t elem_cnt, const T min, const T max,
                   uint32_t random_seed, T* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_LE(min, max);
  std::mt19937 generator(random_seed);
  std::uniform_int_distribution<T> random_distribution(
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
void ConstantInitializer(const T& value, Blob* blob) {
  T* dptr = blob->mut_dptr<T>();
  const int64_t elem_cnt = blob->shape().elem_cnt();
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
void RandomIntUniformInitializer(
    const RandomUniformIntInitializerConf& initializer_conf,
    uint32_t random_seed, Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  RngIntUniform<T>(
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
T GenInitialFan(VarianceNorm variance_norm, Blob* blob,
                const std::string& data_format) {
  int64_t channel_axis = 0;
  if (data_format == "channels_first") {
    channel_axis = 1;
  } else if (data_format == "channels_last") {
    channel_axis = blob->shape().NumAxes() - 1;
  } else {
    UNIMPLEMENTED();
  }
  T fan = ZeroVal<T>::value;
  T fan_in = static_cast<T>(blob->shape().Count(1));
  T fan_out =
      static_cast<T>(blob->shape().Count(0) / blob->shape().At(channel_axis));
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
                       uint32_t random_seed, Blob* blob,
                       const std::string& data_format) {
  CHECK(blob->shape().elem_cnt());
  VarianceNorm variance_norm =
      static_cast<VarianceNorm>(initializer_conf.variance_norm());
  T scale = std::sqrt(static_cast<T>(3)
                      / GenInitialFan<T>(variance_norm, blob, data_format));
  RngUniform<T>(blob->shape().elem_cnt(), static_cast<T>(-scale),
                static_cast<T>(scale), random_seed, blob->mut_dptr<T>());
}

template<typename T>
void MsraInitializer(const MsraInitializerConf& initializer_conf,
                     uint32_t random_seed, Blob* blob,
                     const std::string& data_format) {
  CHECK(blob->shape().elem_cnt());
  VarianceNorm variance_norm =
      static_cast<VarianceNorm>(initializer_conf.variance_norm());
  T std = std::sqrt(static_cast<T>(2)
                    / GenInitialFan<T>(variance_norm, blob, data_format));
  RngNormal<T>(blob->shape().elem_cnt(), ZeroVal<T>::value, static_cast<T>(std),
               random_seed, blob->mut_dptr<T>());
}

void ComputeOffset(const int32_t num_axes, const int64_t* shape,
                   const int32_t* permutation, std::vector<int64_t>& offset) {
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

#define KU_IF_METHOD                     \
  template<typename T, typename Derived> \
  void CpuKernelUtilIf<T, Derived>::

KU_IF_METHOD Axpy(DeviceCtx* ctx, const int n, const T* alpha, const T* x,
                  const int incx, T* y, const int incy) {
  Derived::Axpy(ctx, n, *alpha, x, incx, y, incy);
}
KU_IF_METHOD Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr) {
  *max_ptr = *std::max_element(x, x + n);
}
KU_IF_METHOD Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr,
                 T* temp_storage, size_t temp_storage_bytes) {
  Max(ctx, n, x, max_ptr);
}
KU_IF_METHOD Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr) {
  *sum_ptr = 0;
  for (int64_t i = 0; i < n; ++i) { *sum_ptr += x[i]; }
}
KU_IF_METHOD Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr,
                 T* temp_storage, size_t temp_storage_bytes) {
  Sum(ctx, n, x, sum_ptr);
}
KU_IF_METHOD Transpose(DeviceCtx* ctx, const int32_t num_axis,
                       const Shape& x_shape, const Shape& y_shape,
                       const PbRf<int32_t>& permutation, const int64_t elem_cnt,
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
  ComputeOffset(trans_axis, y_shape.dim_vec().data(), permutation.data(),
                x_to_y_offset);
  std::vector<int64_t> x_index_digits(trans_axis, 0);
  int64_t num_blocks = elem_cnt / block_size;
  FOR_RANGE(int64_t, x_idx, 0, num_blocks) {
    int64_t y_idx =
        std::inner_product(x_to_y_offset.cbegin(), x_to_y_offset.cend(),
                           x_index_digits.cbegin(), 0);
    if (block_size == 1) {
      y[y_idx] = x[x_idx];
    } else {
      memcpy(y + block_size * y_idx, x + block_size * x_idx,
             block_size * sizeof(T));
    }
    IncreaseIndex(x_shape.dim_vec().data(), x_index_digits);
  }
}
KU_IF_METHOD InitializeWithDir(DeviceCtx* ctx, int32_t part_id,
                               int32_t part_num, const std::string& model_dir,
                               Blob* blob, const std::string& bn_in_op,
                               int32_t dim_num, int64_t num_in_each_dim) {
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

#define KU_FLOATING_METHOD             \
  template<typename T>                 \
  void KernelUtil<DeviceType::kCPU, T, \
                  typename std::enable_if<IsFloating<T>::value>::type>::

KU_FLOATING_METHOD Dot(DeviceCtx* ctx, const int n, const T* x, const int incx,
                       const T* y, const int incy, T* result) {
  *result = cblas_dot<T>(n, x, incx, y, incy);
}
KU_FLOATING_METHOD Copy(DeviceCtx* ctx, const int n, const T* x, const int incx,
                        T* y, const int incy) {
  cblas_copy<T>(n, x, incx, y, incy);
}
KU_FLOATING_METHOD Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x,
                        const int incx, T* y, const int incy) {
  cblas_axpy<T>(n, alpha, x, incx, y, incy);
}
KU_FLOATING_METHOD Scal(DeviceCtx* ctx, const int n, const T alpha, T* x,
                        const int incx) {
  cblas_scal<T>(n, alpha, x, incx);
}
KU_FLOATING_METHOD Scal(DeviceCtx* ctx, const int n, const T* alpha, T* x,
                        const int incx) {
  Scal(ctx, n, *alpha, x, incx);
}
KU_FLOATING_METHOD Gemv(DeviceCtx* ctx, const enum CBLAS_TRANSPOSE trans, int m,
                        int n, const T alpha, const T* a, int lda, const T* x,
                        const int incx, const T beta, T* y, const int incy) {
  // Set col major to keep it as the same with cublas
  cblas_gemv<T>(CBLAS_ORDER::CblasColMajor, trans, m, n, alpha, a, lda, x, incx,
                beta, y, incy);
}
KU_FLOATING_METHOD Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                        const enum CBLAS_TRANSPOSE trans_a,
                        const enum CBLAS_TRANSPOSE trans_b, const int m,
                        const int n, const int k, const T alpha, const T* a,
                        const int lda, const T* b, const int ldb, const T beta,
                        T* c, const int ldc) {
  cblas_gemm<T>(order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta,
                c, ldc);
}

KU_FLOATING_METHOD Exp(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  for (int64_t i = 0; i < n; ++i) { y[i] = std::exp(x[i]); }
}
KU_FLOATING_METHOD Div(DeviceCtx* ctx, const int64_t n, T* x, const T* alpha) {
  for (int64_t i = 0; i < n; ++i) { x[i] = x[i] / (*alpha); }
}
KU_FLOATING_METHOD Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y,
                       T* z) {
  for (int64_t i = 0; i < n; ++i) { z[i] = x[i] * y[i]; }
}
KU_FLOATING_METHOD Rsqrt(DeviceCtx* ctx, const int64_t n, T* x,
                         const float epsilon) {
  for (int64_t i = 0; i < n; ++i) { x[i] = 1.0 / std::sqrt(x[i] + epsilon); }
}

KU_FLOATING_METHOD Sigmoid(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  T half = static_cast<T>(0.5);
  for (int64_t i = 0; i != n; ++i) {
    y[i] = half * std::tanh(half * x[i]) + half;
  }
}
KU_FLOATING_METHOD SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x,
                                   const T* y, const T* dy, T* dx) {
  for (int64_t i = 0; i != n; ++i) { dx[i] = y[i] * (1 - y[i]) * dy[i]; }
}
KU_FLOATING_METHOD TanH(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  for (int64_t i = 0; i != n; ++i) { y[i] = std::tanh(x[i]); }
}
KU_FLOATING_METHOD TanHBackward(DeviceCtx* ctx, const int64_t n, const T* x,
                                const T* y, const T* dy, T* dx) {
  for (int64_t i = 0; i != n; ++i) { dx[i] = (1 - y[i] * y[i]) * dy[i]; }
}
KU_FLOATING_METHOD Relu(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  T zero = ZeroVal<T>::value;
  for (int64_t i = 0; i != n; ++i) { y[i] = std::max(x[i], zero); }
}
KU_FLOATING_METHOD ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x,
                                const T* y, const T* dy, T* dx) {
  T zero = ZeroVal<T>::value;
  for (int64_t i = 0; i != n; ++i) { dx[i] = (y[i] > zero) * dy[i]; }
}

KU_FLOATING_METHOD InitializeWithConf(DeviceCtx* ctx,
                                      const InitializerConf& initializer_conf,
                                      uint32_t random_seed, Blob* blob) {
  if (initializer_conf.has_constant_conf()) {
    ConstantInitializer<T>(
        static_cast<T>(initializer_conf.constant_conf().value()), blob);
  } else if (initializer_conf.has_random_uniform_conf()) {
    RandomUniformInitializer<T>(initializer_conf.random_uniform_conf(),
                                random_seed, blob);
  } else if (initializer_conf.has_random_normal_conf()) {
    RandomNormalInitializer<T>(initializer_conf.random_normal_conf(),
                               random_seed, blob);
  } else {
    UNIMPLEMENTED();
  }
}
KU_FLOATING_METHOD InitializeWithConf(DeviceCtx* ctx,
                                      const InitializerConf& initializer_conf,
                                      uint32_t random_seed, Blob* blob,
                                      const std::string& data_format) {
  if (data_format.size() == 0) {
    InitializeWithConf(ctx, initializer_conf, random_seed, blob);
    return;
  }
  if (initializer_conf.has_xavier_conf()) {
    XavierInitializer<T>(initializer_conf.xavier_conf(), random_seed, blob,
                         data_format);
  } else if (initializer_conf.has_msra_conf()) {
    MsraInitializer<T>(initializer_conf.msra_conf(), random_seed, blob,
                       data_format);
  } else {
    InitializeWithConf(ctx, initializer_conf, random_seed, blob);
  }
}

#define KU_INTEGRAL_METHOD             \
  template<typename T>                 \
  void KernelUtil<DeviceType::kCPU, T, \
                  typename std::enable_if<IsIntegral<T>::value>::type>::

KU_INTEGRAL_METHOD Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x,
                        const int incx, T* y, const int incy) {
  FOR_RANGE(int, i, 0, n) {
    *y += alpha * *x;
    x += incx;
    y += incy;
  }
}
KU_INTEGRAL_METHOD InitializeWithConf(DeviceCtx* ctx,
                                      const InitializerConf& initializer_conf,
                                      uint32_t random_seed, Blob* blob) {
  if (initializer_conf.has_constant_int_conf()) {
    ConstantInitializer<T>(
        static_cast<T>(initializer_conf.constant_int_conf().value()), blob);
  } else if (initializer_conf.has_random_uniform_int_conf()) {
    RandomIntUniformInitializer<T>(initializer_conf.random_uniform_int_conf(),
                                   random_seed, blob);
  } else {
    UNIMPLEMENTED();
  }
}
KU_INTEGRAL_METHOD InitializeWithConf(DeviceCtx* ctx,
                                      const InitializerConf& initializer_conf,
                                      uint32_t random_seed, Blob* blob,
                                      const std::string& data_format) {
  InitializeWithConf(ctx, initializer_conf, random_seed, blob);
}

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto)                      \
  template struct CpuKernelUtilIf<type_cpp,                                \
                                  KernelUtil<DeviceType::kCPU, type_cpp>>; \
  template struct KernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

}  //  namespace oneflow
