#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T>
void RngUniform(const int64_t elem_cnt, const T min, const T max, uint32_t random_seed, T* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_LE(min, max);
  std::mt19937 generator(random_seed);
  std::uniform_real_distribution<T> random_distribution(min, std::nextafter(max, GetMaxVal<T>()));
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = random_distribution(generator); }
}

template<typename T>
void RngIntUniform(const int64_t elem_cnt, const T min, const T max, uint32_t random_seed,
                   T* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_LE(min, max);
  std::mt19937 generator(random_seed);
  std::uniform_int_distribution<T> random_distribution(min, std::nextafter(max, GetMaxVal<T>()));
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = random_distribution(generator); }
}

template<typename T>
void RngNormal(const int64_t elem_cnt, const T mean, const T std, uint32_t random_seed, T* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_GT(std, 0.0);
  std::mt19937 generator(random_seed);
  std::normal_distribution<T> random_distribution(mean, std);

  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = random_distribution(generator); }
}

template<>
void RngNormal<float16>(const int64_t elem_cnt, const float16 mean, const float16 std,
                        uint32_t random_seed, float16* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_GT(std, 0.0);
  std::mt19937 generator(random_seed);
  std::normal_distribution<float> random_distribution(mean, std);

  for (int64_t i = 0; i < elem_cnt; ++i) {
    dptr[i] = static_cast<float16>(random_distribution(generator));
  }
}

template<typename T>
void RngTruncatedNormal(const int64_t elem_cnt, const T std, uint32_t random_seed, T* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_GT(std, 0.0);
  T truncated_value = 2 * std;
  std::mt19937 generator(random_seed);
  std::normal_distribution<T> random_distribution(0, std);
  int64_t index = 0;
  while (true) {
    T val = random_distribution(generator);
    if (std::abs(val) < truncated_value) {
      dptr[index++] = val;
      if (index >= elem_cnt) { break; }
    }
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
void RandomUniformInitializer(const RandomUniformInitializerConf& initializer_conf,
                              uint32_t random_seed, Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  RngUniform<T>(blob->shape().elem_cnt(), static_cast<T>(initializer_conf.min()),
                static_cast<T>(initializer_conf.max()), random_seed, blob->mut_dptr<T>());
}

template<typename T>
void RandomIntUniformInitializer(const RandomUniformIntInitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  RngIntUniform<T>(blob->shape().elem_cnt(), static_cast<T>(initializer_conf.min()),
                   static_cast<T>(initializer_conf.max()), random_seed, blob->mut_dptr<T>());
}

template<typename T>
void RandomNormalInitializer(const RandomNormalInitializerConf& initializer_conf,
                             uint32_t random_seed, Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  RngNormal<T>(blob->shape().elem_cnt(), static_cast<T>(initializer_conf.mean()),
               static_cast<T>(initializer_conf.std()), random_seed, blob->mut_dptr<T>());
}

template<typename T>
void TruncatedNormalInitializer(const TruncatedNormalInitializerConf& initializer_conf,
                                uint32_t random_seed, Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  RngTruncatedNormal<T>(blob->shape().elem_cnt(), static_cast<T>(initializer_conf.std()),
                        random_seed, blob->mut_dptr<T>());
}

template<typename T>
T GenInitialFan(VarianceNorm variance_norm, Blob* blob, const std::string& data_format) {
  T fan = ZeroVal<T>::value;
  T fan_in = static_cast<T>(blob->shape().Count(1));
  T fan_out = static_cast<T>(blob->shape().At(0));
  if (data_format == "channels_first") {
    fan_out *= static_cast<T>(blob->shape().Count(2));
  } else if (data_format == "channels_last") {
    fan_out *= static_cast<T>(blob->shape().Count(1, blob->shape().NumAxes() - 1));
  } else {
    CHECK_EQ(blob->shape().NumAxes(), 2);
    CHECK_EQ(data_format, "");
  }
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
void XavierInitializer(const XavierInitializerConf& initializer_conf, uint32_t random_seed,
                       Blob* blob, const std::string& data_format) {
  CHECK(blob->shape().elem_cnt());
  VarianceNorm variance_norm = static_cast<VarianceNorm>(initializer_conf.variance_norm());
  T scale = std::sqrt(static_cast<T>(3) / GenInitialFan<T>(variance_norm, blob, data_format));
  RngUniform<T>(blob->shape().elem_cnt(), static_cast<T>(-scale), static_cast<T>(scale),
                random_seed, blob->mut_dptr<T>());
}

template<typename T>
void MsraInitializer(const MsraInitializerConf& initializer_conf, uint32_t random_seed, Blob* blob,
                     const std::string& data_format) {
  CHECK(blob->shape().elem_cnt());
  VarianceNorm variance_norm = static_cast<VarianceNorm>(initializer_conf.variance_norm());
  T std = std::sqrt(static_cast<T>(2) / GenInitialFan<T>(variance_norm, blob, data_format));
  RngNormal<T>(blob->shape().elem_cnt(), ZeroVal<T>::value, static_cast<T>(std), random_seed,
               blob->mut_dptr<T>());
}

template<typename T>
void RangeInitializer(int64_t outer_size, int64_t idx_dim_size, int64_t inner_size, T start,
                      T stride, T* out) {
  FOR_RANGE(int64_t, i, 0, outer_size) {
    FOR_RANGE(int64_t, j, 0, idx_dim_size) {
      FOR_RANGE(int64_t, k, 0, inner_size) {
        *(out + i * idx_dim_size * inner_size + j * inner_size + k) = start + j * stride;
      }
    }
  }
}

template<typename T, typename RangeInitializerConfT>
void RangeInitializer(const RangeInitializerConfT& initializer_conf, uint32_t random_seed,
                      Blob* blob) {
  CHECK_GT(blob->shape().NumAxes(), 0);
  const int64_t axis = initializer_conf.axis() < 0
                           ? blob->shape().NumAxes() + initializer_conf.axis()
                           : initializer_conf.axis();
  CHECK_GE(axis, 0);
  CHECK_LT(axis, blob->shape().NumAxes());
  RangeInitializer<T>(blob->shape().Count(0, axis), blob->shape().At(axis),
                      blob->shape().Count(axis + 1), static_cast<T>(initializer_conf.start()),
                      static_cast<T>(initializer_conf.stride()), blob->mut_dptr<T>());
}

template<typename T>
void RangeInitializer(const RangeInitializerConf& initializer_conf, uint32_t random_seed,
                      Blob* blob) {
  RangeInitializer<T, RangeInitializerConf>(initializer_conf, random_seed, blob);
}

template<typename T>
void IntSequenceInitializer(const IntRangeInitializerConf& initializer_conf, uint32_t random_seed,
                            Blob* blob) {
  RangeInitializer<T, IntRangeInitializerConf>(initializer_conf, random_seed, blob);
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

template<typename T, T (*reduce_core_func)(const T, const T)>
void MatrixRowReduce(const int64_t row_num, const int64_t col_num, const T* x, T* y) {
  FOR_RANGE(int64_t, i, 0, row_num) {
    y[i] = x[i * col_num];
    FOR_RANGE(int64_t, j, 1, col_num) { y[i] = reduce_core_func(y[i], x[i * col_num + j]); }
  }
}

template<typename T>
void InitializeWithDirCpu(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                          const std::string& model_dir, Blob* blob, const std::string& bn_in_op,
                          int32_t dim_num, int64_t num_in_each_dim) {
  int64_t blob_size = blob->ByteSizeOfDataContentField();
  int64_t byte_size_of_each_dim = num_in_each_dim * sizeof(T);
  std::string file_path = JoinPath(model_dir, bn_in_op);
  uint64_t file_size = SnapshotFS()->GetFileSize(file_path);
  CHECK_EQ(file_size, dim_num * byte_size_of_each_dim);
  BalancedSplitter splitter = BalancedSplitter(dim_num, part_num);
  int64_t begin_pos = splitter.At(part_id).begin() * byte_size_of_each_dim;
  PersistentInStream in_stream(SnapshotFS(), file_path, begin_pos);
  in_stream.Read(blob->mut_dptr<char>(), blob_size);
}

}  // namespace

// CPU && Floating
template<typename T>
struct NewKernelUtilIf<DeviceType::kCPU, T, typename std::enable_if<IsFloating<T>::value>::type> {
  static void OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const T alpha, const T* a, const T* b,
                     const T beta, T* c) {
    const int lda = (trans_a == CblasNoTrans) ? k : m;
    const int ldb = (trans_b == CblasNoTrans) ? n : k;
    const int ldc = n;

    FloatingNewKernelUtilIf<DeviceType::kCPU, T>::Gemm(ctx, CblasRowMajor, trans_a, trans_b, m, n,
                                                       k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob, const std::string& data_format) {
    if (initializer_conf.has_constant_conf()) {
      ConstantInitializer<T>(static_cast<T>(initializer_conf.constant_conf().value()), blob);
    } else if (initializer_conf.has_random_uniform_conf()) {
      RandomUniformInitializer<T>(initializer_conf.random_uniform_conf(), random_seed, blob);
    } else if (initializer_conf.has_random_normal_conf()) {
      RandomNormalInitializer<T>(initializer_conf.random_normal_conf(), random_seed, blob);
    } else if (initializer_conf.has_truncated_normal_conf()) {
      TruncatedNormalInitializer<T>(initializer_conf.truncated_normal_conf(), random_seed, blob);
    } else if (initializer_conf.has_xavier_conf()) {
      XavierInitializer<T>(initializer_conf.xavier_conf(), random_seed, blob, data_format);
    } else if (initializer_conf.has_msra_conf()) {
      MsraInitializer<T>(initializer_conf.msra_conf(), random_seed, blob, data_format);
    } else if (initializer_conf.has_range_conf()) {
      RangeInitializer<T>(initializer_conf.range_conf(), random_seed, blob);
    } else {
      UNIMPLEMENTED();
    }
  }
  static void InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                                const std::string& model_dir, Blob* blob,
                                const std::string& bn_in_op, int32_t dim_num,
                                int64_t num_in_each_dim) {
    InitializeWithDirCpu<T>(ctx, part_id, part_num, model_dir, blob, bn_in_op, dim_num,
                            num_in_each_dim);
  }
  static void Sigmoid(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    T half = static_cast<T>(0.5);
    for (int64_t i = 0; i != n; ++i) { y[i] = half * std::tanh(half * x[i]) + half; }
  }
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                              T* dx) {
    for (int64_t i = 0; i != n; ++i) { dx[i] = y[i] * (1 - y[i]) * dy[i]; }
  }
  static void TanH(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    for (int64_t i = 0; i != n; ++i) { y[i] = std::tanh(x[i]); }
  }
  static void TanHBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx) {
    for (int64_t i = 0; i != n; ++i) { dx[i] = (1 - y[i] * y[i]) * dy[i]; }
  }
  static void Relu(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    T zero = ZeroVal<T>::value;
    for (int64_t i = 0; i != n; ++i) { y[i] = std::max(x[i], zero); }
  }
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx) {
    T zero = ZeroVal<T>::value;
    for (int64_t i = 0; i != n; ++i) { dx[i] = (y[i] > zero) * dy[i]; }
  }
  static void Set(DeviceCtx* ctx, const T value, T* addr) { *addr = value; }
};

// CPU && Integral
template<typename T>
struct NewKernelUtilIf<DeviceType::kCPU, T, typename std::enable_if<IsIntegral<T>::value>::type> {
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob, const std::string& data_format) {
    if (initializer_conf.has_constant_int_conf()) {
      ConstantInitializer<T>(static_cast<T>(initializer_conf.constant_int_conf().value()), blob);
    } else if (initializer_conf.has_random_uniform_int_conf()) {
      RandomIntUniformInitializer<T>(initializer_conf.random_uniform_int_conf(), random_seed, blob);
    } else if (initializer_conf.has_int_range_conf()) {
      IntSequenceInitializer<T>(initializer_conf.int_range_conf(), random_seed, blob);
    } else {
      UNIMPLEMENTED();
    }
  }
  static void InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                                const std::string& model_dir, Blob* blob,
                                const std::string& bn_in_op, int32_t dim_num,
                                int64_t num_in_each_dim) {
    InitializeWithDirCpu<T>(ctx, part_id, part_num, model_dir, blob, bn_in_op, dim_num,
                            num_in_each_dim);
  }
  static void Set(DeviceCtx* ctx, const T value, T* addr) { *addr = value; }
};

// CPU && Float16
template<typename T>
struct NewKernelUtilIf<DeviceType::kCPU, T, typename std::enable_if<IsFloat16<T>::value>::type> {
  static void OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const T alpha, const T* a, const T* b,
                     const T beta, T* c) {
    const int lda = (trans_a == CblasNoTrans) ? k : m;
    const int ldb = (trans_b == CblasNoTrans) ? n : k;
    const int ldc = n;

    Float16NewKernelUtilIf<DeviceType::kCPU, T>::HGemm(ctx, CblasRowMajor, trans_a, trans_b, m, n,
                                                       k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob, const std::string& data_format) {
    // TODO: do float16 support all the initializer conf below?
    if (initializer_conf.has_constant_conf()) {
      ConstantInitializer<T>(static_cast<T>(initializer_conf.constant_conf().value()), blob);
    } else if (initializer_conf.has_random_uniform_conf()) {
      // RandomUniformInitializer<T>(initializer_conf.random_uniform_conf(), random_seed, blob);
    } else if (initializer_conf.has_random_normal_conf()) {
      RandomNormalInitializer<T>(initializer_conf.random_normal_conf(), random_seed, blob);
    } else if (initializer_conf.has_range_conf()) {
      RangeInitializer<T>(initializer_conf.range_conf(), random_seed, blob);
    } else {
      UNIMPLEMENTED();
    }
  }
  static void InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                                const std::string& model_dir, Blob* blob,
                                const std::string& bn_in_op, int32_t dim_num,
                                int64_t num_in_each_dim) {
    InitializeWithDirCpu<T>(ctx, part_id, part_num, model_dir, blob, bn_in_op, dim_num,
                            num_in_each_dim);
  }
  static void Sigmoid(DeviceCtx* ctx, const int64_t n, const T* x, T* y) { UNIMPLEMENTED(); }
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                              T* dx) {
    UNIMPLEMENTED();
  }
  static void TanH(DeviceCtx* ctx, const int64_t n, const T* x, T* y) { UNIMPLEMENTED(); }
  static void TanHBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx) {
    UNIMPLEMENTED();
  }
  static void Relu(DeviceCtx* ctx, const int64_t n, const T* x, T* y) { UNIMPLEMENTED(); }
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx) {
    UNIMPLEMENTED();
  }
  static void Set(DeviceCtx* ctx, const T value, T* addr) { *addr = value; }
};

template<typename T>
struct FloatingNewKernelUtilIf<DeviceType::kCPU, T> {
  static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
                   const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                   const T alpha, const T* a, const int lda, const T* b, const int ldb,
                   const T beta, T* c, const int ldc) {
    cblas_gemm<T>(order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
};

template<typename T>
struct Float16NewKernelUtilIf<DeviceType::kCPU, T> {
  static void HGemm(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                    const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                    const int m, const int n, const int k, const T alpha, const T* a, const int lda,
                    const T* b, const int ldb, const T beta, T* c, const int ldc) {
    UNIMPLEMENTED();
  }
  static void Half2Float(DeviceCtx* ctx, const int n, const T* src, float* dst) {
    for (size_t i = 0; i < n; ++i) { dst[i] = static_cast<float>(src[i]); }
  }
  static void Float2Half(DeviceCtx* ctx, const int n, const float* src, T* dst) {
    for (size_t i = 0; i < n; ++i) { dst[i] = static_cast<float16>(src[i]); }
  }
};

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct NewKernelUtilIf<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);

#define INSTANTIATE_FLOATING_KERNEL_UTIL(type_cpp, type_proto) \
  template struct FloatingNewKernelUtilIf<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_FLOATING_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

#define INSTANTIATE_FLOAT16_KERNEL_UTIL(type_cpp, type_proto) \
  template struct Float16NewKernelUtilIf<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_FLOAT16_KERNEL_UTIL, FLOAT16_DATA_TYPE_SEQ);

}  //  namespace oneflow
