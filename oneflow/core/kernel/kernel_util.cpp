#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename FloatingPointType>
static void RngUniform(const int64_t elem_cnt, const FloatingPointType min,
                       const FloatingPointType max, FloatingPointType* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_LE(min, max);
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<FloatingPointType> random_distribution(
      min, std::nextafter(max, std::numeric_limits<FloatingPointType>::max()));

  for (size_t i = 0; i < elem_cnt; ++i) {
    dptr[i] = random_distribution(generator);
  }
}

template<typename FloatingPointType>
static void RngGaussian(const int64_t elem_cnt, const FloatingPointType mean,
                        const FloatingPointType std, FloatingPointType* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_GT(std, 0);
  std::random_device rd;
  std::mt19937 generator(rd());
  std::normal_distribution<FloatingPointType> random_distribution(mean, std);

  for (size_t i = 0; i < elem_cnt; i++) {
    dptr[i] = random_distribution(generator);
  }
}

template<typename FloatingPointType>
static void RngBernoulli(const int64_t elem_cnt,
                         const FloatingPointType non_zero_probability,
                         bool* mask) {
  CHECK_GE(elem_cnt, 0);
  CHECK(mask);
  CHECK_GE(non_zero_probability, 0);
  CHECK_LE(non_zero_probability, 1);
  std::random_device rd;
  std::mt19937 generator(rd());
  std::bernoulli_distribution random_distribution(non_zero_probability);

  for (size_t i = 0; i < elem_cnt; i++) {
    mask[i] = random_distribution(generator);
  }
}

}  // namespace

template<typename FloatingPointType>
class KernelUtil<DeviceType::kCPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelUtil);
  KernelUtil() = delete;

  static void Memcpy(
      const KernelCtx& ctx, void* dst, const void* src, size_t sz,
      cudaMemcpyKind kind = cudaMemcpyKind::cudaMemcpyHostToHost) {
    ctx.device_ctx->cpu_stream()->SendWork(
        [dst, src, sz]() { memcpy(dst, src, sz); });
  }

  static void Memset(const KernelCtx& ctx, void* dst, const char value,
                     size_t sz) {
    ctx.device_ctx->cpu_stream()->SendWork(
        [dst, value, sz]() { memset(dst, value, sz); });
  }

  static void BlasAxpy(const KernelCtx& ctx, const int n,
                       const FloatingPointType alpha,
                       const FloatingPointType* x, const int incx,
                       FloatingPointType* y, const int incy) {
    ctx.device_ctx->cpu_stream()->SendWork([n, alpha, x, incx, y, incy]() {
      cblas_axpy(n, alpha, x, incx, y, incy);
    });
  }

  static void BlasScal(const KernelCtx& ctx, const int n,
                       const FloatingPointType alpha, FloatingPointType* x,
                       const int incx) {
    ctx.device_ctx->cpu_stream()->SendWork(
        [n, alpha, x, incx]() { cblas_scal(n, alpha, x, incx); });
  }

  static void BlasGemv(const KernelCtx& ctx, const enum CBLAS_TRANSPOSE trans,
                       int m, int n, const FloatingPointType alpha,
                       const FloatingPointType* a, int lda,
                       const FloatingPointType* x, const int incx,
                       const FloatingPointType beta, FloatingPointType* y,
                       const int incy) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      // Set col major to keep it as the same with cublas
      cblas_gemv(CBLAS_ORDER::CblasColMajor, trans, m, n, alpha, a, lda, x,
                 incx, beta, y, incy);
    });
  }

  static void BlasGemm(const KernelCtx& ctx, const enum CBLAS_ORDER order,
                       const enum CBLAS_TRANSPOSE trans_a,
                       const enum CBLAS_TRANSPOSE trans_b, const int m,
                       const int n, const int k, const FloatingPointType alpha,
                       const FloatingPointType* a, const int lda,
                       const FloatingPointType* b, const int ldb,
                       const FloatingPointType beta, FloatingPointType* c,
                       const int ldc) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      cblas_gemm(order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
    });
  }

  static void BlasDot(const KernelCtx& ctx, const int n,
                      const FloatingPointType* x, const int incx,
                      const FloatingPointType* y, const int incy,
                      FloatingPointType* result) {
    ctx.device_ctx->cpu_stream()->SendWork(
        [=]() { *result = cblas_dot(n, x, incx, y, incy); });
  }

  static void BlasSwap(const KernelCtx& ctx, const int n, FloatingPointType* x,
                       const int incx, FloatingPointType* y, const int incy) {
    ctx.device_ctx->cpu_stream()->SendWork(
        [=]() { cblas_swap(n, x, incx, y, incy); });
  }

  static void BlasCopy(const KernelCtx& ctx, const int n,
                       const FloatingPointType* x, const int incx,
                       FloatingPointType* y, const int incy) {
    ctx.device_ctx->cpu_stream()->SendWork(
        [=]() { cblas_copy(n, x, incx, y, incy); });
  }

  static void Filler(const KernelCtx& ctx, const FillerConf& filler_conf,
                     Blob* blob) {
    if (filler_conf.has_constant_conf()) {
      ConstantFiller(ctx, dynamic_cast<const ConstantFillerConf&>(filler_conf),
                     blob);
    } else if (filler_conf.has_uniform_conf()) {
      UniformFiller(ctx, dynamic_cast<const UniformFillerConf&>(filler_conf),
                    blob);
    } else if (filler_conf.has_gaussian_conf()) {
      GaussianFiller(ctx, dynamic_cast<const GaussianFillerConf&>(filler_conf),
                     blob);
    } else {
      CHECK(false) << "Unknown filler name";
    }
  }

 private:
  static void ConstantFiller(const KernelCtx& ctx,
                             const ConstantFillerConf& filler_conf,
                             Blob* blob) {
    FloatingPointType* dptr = static_cast<FloatingPointType*>(blob->mut_dptr());
    const int64_t elem_cnt = blob->shape().elem_cnt();
    const FloatingPointType value = filler_conf.value();
    CHECK(elem_cnt);
    for (size_t i = 0; i < elem_cnt; ++i) { dptr[i] = value; }
  }

  static void UniformFiller(const KernelCtx& ctx,
                            const UniformFillerConf& filler_conf, Blob* blob) {
    CHECK(blob->shape().elem_cnt());
    ctx.device_ctx->cpu_stream()->Send([=]() {
      RngUniform<FloatingPointType>(
          blob->shape().elem_cnt(),
          static_cast<FloatingPointType>(filler_conf.min()),
          static_cast<FloatingPointType>(filler_conf.max()),
          static_cast<FloatingPointType*>(blob->mut_dptr()));
    });
  }

  static void GaussianFiller(const KernelCtx& ctx,
                             const GaussianFillerConf& filler_conf,
                             Blob* blob) {
    CHECK(blob->shape().elem_cnt());
    ctx.device_ctx->cpu_stream()->Send([=]() {
      RngGaussian<FloatingPointType>(
          blob->shape().elem_cnt(),
          static_cast<FloatingPointType>(filler_conf.mean()),
          static_cast<FloatingPointType>(filler_conf.std()),
          static_cast<FloatingPointType*>(blob->mut_dptr()));
    });
  }
};

INSTANTIATE_CPU_KERNEL_UTIL_CLASS(KernelUtil);

}  //  namespace oneflow
