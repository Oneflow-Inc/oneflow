#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename FloatingPointType>
void RngUniform(const int64_t elem_cnt, const FloatingPointType min,
                const FloatingPointType max, uint32_t random_seed,
                FloatingPointType* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_LE(min, max);
  std::mt19937 generator(random_seed);
  std::uniform_real_distribution<FloatingPointType> random_distribution(
      min, std::nextafter(max, std::numeric_limits<FloatingPointType>::max()));

  for (int64_t i = 0; i < elem_cnt; ++i) {
    dptr[i] = random_distribution(generator);
  }
}

template<typename FloatingPointType>
void RngGaussian(const int64_t elem_cnt, const FloatingPointType mean,
                 const FloatingPointType std, uint32_t random_seed,
                 FloatingPointType* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_GT(std, 0.0);
  std::mt19937 generator(random_seed);
  std::normal_distribution<FloatingPointType> random_distribution(mean, std);

  for (int64_t i = 0; i < elem_cnt; ++i) {
    dptr[i] = random_distribution(generator);
  }
}

}  // namespace

template<typename FloatingPointType>
class KernelUtil<DeviceType::kCPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelUtil);
  KernelUtil() = delete;

  static void BlasAxpy(DeviceCtx* ctx, const int n,
                       const FloatingPointType alpha,
                       const FloatingPointType* x, const int incx,
                       FloatingPointType* y, const int incy) {
    ctx->cpu_stream()->SendWork([n, alpha, x, incx, y, incy]() {
      cblas_axpy(n, alpha, x, incx, y, incy);
    });
  }

  static void BlasScal(DeviceCtx* ctx, const int n,
                       const FloatingPointType alpha, FloatingPointType* x,
                       const int incx) {
    ctx->cpu_stream()->SendWork(
        [n, alpha, x, incx]() { cblas_scal(n, alpha, x, incx); });
  }

  static void Max(DeviceCtx* ctx, const int64_t n, const FloatingPointType* x,
                  FloatingPointType* max_ptr) {
    Max(ctx, n, x, max_ptr, nullptr, 0);
  }

  static void Max(DeviceCtx* ctx, const int64_t n, const FloatingPointType* x,
                  FloatingPointType* max_ptr, FloatingPointType* temp_storage,
                  size_t temp_storage_bytes) {
    ctx->cpu_stream()->SendWork([=]() {
      *max_ptr = x[0];
      for (int64_t i = 0; i < n; ++i) { *max_ptr = std::max(*max_ptr, x[i]); }
    });
  }

  static void Exp(DeviceCtx* ctx, const int64_t n, const FloatingPointType* x,
                  FloatingPointType* y) {
    ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) { y[i] = std::exp(x[i]); }
    });
  }

  static void Sum(DeviceCtx* ctx, const int64_t n, const FloatingPointType* x,
                  FloatingPointType* sum_ptr) {
    Sum(ctx, n, x, sum_ptr, nullptr, 0);
  }

  static void Sum(DeviceCtx* ctx, const int64_t n, const FloatingPointType* x,
                  FloatingPointType* sum_ptr, FloatingPointType* temp_storage,
                  size_t temp_storage_bytes) {
    ctx->cpu_stream()->SendWork([=]() {
      *sum_ptr = 0;
      for (int64_t i = 0; i < n; ++i) { *sum_ptr += x[i]; }
    });
  }

  static void Div(DeviceCtx* ctx, const int64_t n, FloatingPointType* x,
                  const FloatingPointType* alpha_ptr) {
    ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) { x[i] = x[i] / (*alpha_ptr); }
    });
  }

  static void Mul(DeviceCtx* ctx, const int64_t n, const FloatingPointType* x,
                  const FloatingPointType* y, FloatingPointType* z) {
    ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) { z[i] = x[i] * y[i]; }
    });
  }

  static void BlasGemv(DeviceCtx* ctx, const enum CBLAS_TRANSPOSE trans, int m,
                       int n, const FloatingPointType alpha,
                       const FloatingPointType* a, int lda,
                       const FloatingPointType* x, const int incx,
                       const FloatingPointType beta, FloatingPointType* y,
                       const int incy) {
    ctx->cpu_stream()->SendWork([=]() {
      // Set col major to keep it as the same with cublas
      cblas_gemv(CBLAS_ORDER::CblasColMajor, trans, m, n, alpha, a, lda, x,
                 incx, beta, y, incy);
    });
  }

  static void BlasGemm(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                       const enum CBLAS_TRANSPOSE trans_a,
                       const enum CBLAS_TRANSPOSE trans_b, const int m,
                       const int n, const int k, const FloatingPointType alpha,
                       const FloatingPointType* a, const int lda,
                       const FloatingPointType* b, const int ldb,
                       const FloatingPointType beta, FloatingPointType* c,
                       const int ldc) {
    ctx->cpu_stream()->SendWork([=]() {
      cblas_gemm(order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
    });
  }

  static void BlasDot(DeviceCtx* ctx, const int n, const FloatingPointType* x,
                      const int incx, const FloatingPointType* y,
                      const int incy, FloatingPointType* result) {
    ctx->cpu_stream()->SendWork(
        [=]() { *result = cblas_dot(n, x, incx, y, incy); });
  }

  static void BlasSwap(DeviceCtx* ctx, const int n, FloatingPointType* x,
                       const int incx, FloatingPointType* y, const int incy) {
    ctx->cpu_stream()->SendWork([=]() { cblas_swap(n, x, incx, y, incy); });
  }

  static void BlasCopy(DeviceCtx* ctx, const int n, const FloatingPointType* x,
                       const int incx, FloatingPointType* y, const int incy) {
    ctx->cpu_stream()->SendWork([=]() { cblas_copy(n, x, incx, y, incy); });
  }

  static void Fill(const FillConf& fill_conf, uint32_t random_seed,
                   Blob* blob) {
    if (fill_conf.has_constant_conf()) {
      ConstantFill(fill_conf.constant_conf(), blob);
    } else if (fill_conf.has_uniform_conf()) {
      UniformFill(fill_conf.uniform_conf(), random_seed, blob);
    } else if (fill_conf.has_gaussian_conf()) {
      GaussianFill(fill_conf.gaussian_conf(), random_seed, blob);
    } else {
      UNEXPECTED_RUN();
    }
  }

  static void Fill(DeviceCtx* ctx, const FillConf& fill_conf,
                   uint32_t random_seed, Blob* blob) {
    ctx->cpu_stream()->SendWork([=]() { Fill(fill_conf, random_seed, blob); });
  }

  static void FillWithSnapshot(DeviceCtx* ctx, int32_t part_id,
                               int32_t part_num, const Snapshot* snapshot,
                               Blob* blob, const std::string& lbn,
                               int32_t dim_num, int64_t num_in_each_dim) {
    int64_t blob_size = blob->shape().elem_cnt() * sizeof(FloatingPointType);
    ctx->cpu_stream()->SendWork([=]() {
      std::unique_ptr<PersistentInStream> in_stream =
          snapshot->GetInStream(lbn, part_id, part_num, dim_num,
                                num_in_each_dim * sizeof(FloatingPointType));
      in_stream->Read(blob->mut_dptr<char>(), blob_size);
    });
  }

 private:
  static void ConstantFill(const ConstantFillConf& fill_conf, Blob* blob) {
    FloatingPointType* dptr = blob->mut_dptr<FloatingPointType>();
    const int64_t elem_cnt = blob->shape().elem_cnt();
    const FloatingPointType value = fill_conf.value();
    CHECK(elem_cnt);
    for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = value; }
  }

  static void UniformFill(const UniformFillConf& fill_conf,
                          uint32_t random_seed, Blob* blob) {
    CHECK(blob->shape().elem_cnt());
    RngUniform<FloatingPointType>(
        blob->shape().elem_cnt(),
        static_cast<FloatingPointType>(fill_conf.min()),
        static_cast<FloatingPointType>(fill_conf.max()), random_seed,
        blob->mut_dptr<FloatingPointType>());
  }

  static void GaussianFill(const GaussianFillConf& fill_conf,
                           uint32_t random_seed, Blob* blob) {
    CHECK(blob->shape().elem_cnt());
    RngGaussian<FloatingPointType>(
        blob->shape().elem_cnt(),
        static_cast<FloatingPointType>(fill_conf.mean()),
        static_cast<FloatingPointType>(fill_conf.std()), random_seed,
        blob->mut_dptr<FloatingPointType>());
  }
};

template class KernelUtil<DeviceType::kCPU, float>;
template class KernelUtil<DeviceType::kCPU, double>;

template<>
void Memcpy<DeviceType::kCPU>(DeviceCtx* ctx, void* dst, const void* src,
                              size_t sz, cudaMemcpyKind kind) {
  ctx->cpu_stream()->SendWork([dst, src, sz]() { memcpy(dst, src, sz); });
}

template<>
void Memset<DeviceType::kCPU>(DeviceCtx* ctx, void* dst, const char value,
                              size_t sz) {
  ctx->cpu_stream()->SendWork([dst, value, sz]() { memset(dst, value, sz); });
}

}  //  namespace oneflow
