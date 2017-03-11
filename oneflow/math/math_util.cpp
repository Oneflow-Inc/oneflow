#include <limits>
#include "math/math_util.h"

#define __builtin_popcount __popcnt
#define __builtin_popcountl __popcnt

namespace caffe {
template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);


template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
int caffe_cpu_hamming_distance<float>(const int n, const float* x,
                                  const float* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcount(static_cast<uint32_t>(x[i]) ^
                               static_cast<uint32_t>(y[i]));
  }
  return dist;
}

template <>
int caffe_cpu_hamming_distance<double>(const int n, const double* x,
                                   const double* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcountl(static_cast<uint64_t>(x[i]) ^
                                static_cast<uint64_t>(y[i]));
  }
  return dist;
}

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}
// rng generating functions
template <>
void caffe_rng_uniform(const int n, const float minimum, const float maximum,
  float* dev) {
  CHECK_GE(n, 0);
  CHECK(dev);
  CHECK_LE(minimum, maximum);
  float *host = reinterpret_cast<float*>(calloc(n, sizeof(float)));
  CHECK(host);
  CURAND_CHECK(curandGenerateUniform(RNG::generator(), host, n));
  if (minimum != 0 || maximum != 1) {
    float alpha = maximum - minimum;
    for (int i = 0; i < n; ++i) {
      host[i] = alpha*host[i] + minimum;
    }
  }
  CUDA_CHECK(cudaMemcpy(dev, host, n*sizeof(float), cudaMemcpyHostToDevice));
  free(host);
}
template <>
void caffe_rng_uniform(const int n, const double minimum, const double maximum,
  double* dev) {
  CHECK_GE(n, 0);
  CHECK(dev);
  CHECK_LE(minimum, maximum);
  double *host = reinterpret_cast<double*>(calloc(n, sizeof(double)));
  CHECK(host);
  CURAND_CHECK(curandGenerateUniformDouble(RNG::generator(), host, n));
  if (minimum != 0 || maximum != 1) {
    double alpha = maximum - minimum;
    for (int i = 0; i < n; ++i) {
      host[i] = alpha*host[i] + minimum;
    }
  }
  CUDA_CHECK(cudaMemcpy(dev, host, n*sizeof(double), cudaMemcpyHostToDevice));
  free(host);
}
void caffe_rng_discrete_uniform(const int n, const float minimum,
  const float maximum, float* dev) {
  CHECK_GE(n, 0);
  CHECK(dev);
  CHECK_LE(minimum, maximum);
  float *host = reinterpret_cast<float*>(calloc(n, sizeof(float)));
  CHECK(host);
  CURAND_CHECK(curandGenerateUniform(RNG::generator(), host, n));
  if (minimum != 0 || maximum != 1) {
    float alpha = maximum - minimum;
    for (int i = 0; i < n; ++i) {
      host[i] = floor(alpha*host[i] + minimum);
    }
  }
  CUDA_CHECK(cudaMemcpy(dev, host, n*sizeof(float), cudaMemcpyHostToDevice));
  free(host);
}
template <>
void caffe_rng_discrete_uniform(const int n, const double minimum,
  const double maximum, double* dev) {
  CHECK_GE(n, 0);
  CHECK(dev);
  CHECK_LE(minimum, maximum);
  double *host = reinterpret_cast<double*>(calloc(n, sizeof(double)));
  CHECK(host);
  CURAND_CHECK(curandGenerateUniformDouble(RNG::generator(), host, n));
  if (minimum != 0 || maximum != 1) {
    double alpha = maximum - minimum;
    for (int i = 0; i < n; ++i) {
      host[i] = floor(alpha*host[i] + minimum);
    }
  }
  CUDA_CHECK(cudaMemcpy(dev, host, n*sizeof(double), cudaMemcpyHostToDevice));
  free(host);
}
template <>
void caffe_rng_gaussian(const int n, const float mean, const float stddev,
  float* dev) {
  CHECK_GE(n, 0);
  CHECK(dev);
  CHECK_GT(stddev, 0);
  float* host = reinterpret_cast<float*>(calloc((n % 2 ? n + 1 : n),
    sizeof(float)));
  CHECK(host);
  CURAND_CHECK(curandGenerateNormal(RNG::generator(), host,
    (n % 2 ? n + 1 : n), mean, stddev));
  CUDA_CHECK(cudaMemcpy(dev, host, n*sizeof(float), cudaMemcpyHostToDevice));
  free(host);
}
template <>
void caffe_rng_gaussian(const int n, const double mean, const double stddev,
  double* dev) {
  CHECK_GE(n, 0);
  CHECK(dev);
  CHECK_GT(stddev, 0);
  double *host = reinterpret_cast<double*>(calloc((n % 2 ? n + 1 : n),
    sizeof(double)));
  CHECK(host);
  CURAND_CHECK(curandGenerateNormalDouble(RNG::generator(), host,
    (n % 2 ? n + 1 : n), mean, stddev));
  CUDA_CHECK(cudaMemcpy(dev, host, n*sizeof(double), cudaMemcpyHostToDevice));
  free(host);
}
template<>
void caffe_rng_bernoulli(const int n, const float non_zero_probability,
  float* dev) {
  CHECK_GE(n, 0);
  CHECK(dev);
  CHECK_GE(non_zero_probability, 0);
  CHECK_LE(non_zero_probability, 1);
  float *host = reinterpret_cast<float*>(calloc(n, sizeof(float)));
  CHECK(host);
  CURAND_CHECK(curandGenerateUniform(RNG::generator(), host, n));
  for (int i = 0; i < n; ++i) {
    if (host[i] >= non_zero_probability)
      host[i] = 0;
    else
      host[i] = 1;
  }
  CUDA_CHECK(cudaMemcpy(dev, host, n*sizeof(float), cudaMemcpyHostToDevice));
  free(host);
}
template<>
void caffe_rng_bernoulli(const int n, const double non_zero_probability,
  double* dev) {
  CHECK_GE(n, 0);
  CHECK(dev);
  CHECK_GE(non_zero_probability, 0);
  CHECK_LE(non_zero_probability, 1);
  double *host = reinterpret_cast<double*>(calloc(n, sizeof(double)));
  CHECK(host);
  CURAND_CHECK(curandGenerateUniformDouble(RNG::generator(), host, n));
  for (int i = 0; i < n; ++i) {
    if (host[i] >= non_zero_probability)
      host[i] = 0;
    else
      host[i] = 1;
  }
  CUDA_CHECK(cudaMemcpy(dev, host, n*sizeof(double), cudaMemcpyHostToDevice));
  free(host);
}
template <>
void caffe_rng_positive_unitball(const int count, const int num, const int dim,
  float* dev) {
  CHECK_GE(count, 0);
  CHECK(dev);
  float *host = reinterpret_cast<float*>(calloc(count, sizeof(float)));
  CHECK(host);
  CURAND_CHECK(curandGenerateUniform(RNG::generator(), host, count));
  for (int i = 0; i < num; ++i) {
    float sum = 0;
    for (int j = 0; j < dim; ++j) {
      sum += host[i*dim+j];
    }
    CHECK_NE(sum, 0);
    for (int j = 0; j < dim; ++j) {
      host[i*dim+j] /= sum;
    }
  }
  CUDA_CHECK(cudaMemcpy(dev, host, count*sizeof(float),
    cudaMemcpyHostToDevice));
  free(host);
}
template <>
void caffe_rng_positive_unitball(const int count, const int num, const int dim,
  double* dev) {
  CHECK_GE(count, 0);
  CHECK(dev);
  double *host = reinterpret_cast<double*>(calloc(count, sizeof(double)));
  CHECK(host);
  CURAND_CHECK(curandGenerateUniformDouble(RNG::generator(), host, count));
  for (int i = 0; i < num; ++i) {
    double sum = 0;
    for (int j = 0; j < dim; ++j) {
      sum += host[i*dim + j];
    }
    CHECK_NE(sum, 0);
    for (int j = 0; j < dim; ++j) {
      host[i*dim + j] /= sum;
    }
  }
  CUDA_CHECK(cudaMemcpy(dev, host, count*sizeof(double),
    cudaMemcpyHostToDevice));
  free(host);
}
}  // namespace caffe
