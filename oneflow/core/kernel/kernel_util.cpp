#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

uint32_t NewRandomSeed() {
  static std::mt19937 gen{std::random_device{}()};
  return gen();
}

template<typename FloatingPointType>
void RngUniform(const int64_t elem_cnt, const FloatingPointType min,
                const FloatingPointType max, FloatingPointType* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_LE(min, max);
  std::mt19937 generator(NewRandomSeed());
  std::uniform_real_distribution<FloatingPointType> random_distribution(
      min, std::nextafter(max, std::numeric_limits<FloatingPointType>::max()));

  for (int64_t i = 0; i < elem_cnt; ++i) {
    dptr[i] = random_distribution(generator);
  }
}

template<typename FloatingPointType>
void RngGaussian(const int64_t elem_cnt, const FloatingPointType mean,
                 const FloatingPointType std, FloatingPointType* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_GT(std, 0.0);
  std::mt19937 generator(NewRandomSeed());
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

  static void Max(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, FloatingPointType* max_ptr) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      *max_ptr = x[0];
      for (int64_t i = 0; i < n; ++i) { *max_ptr = std::max(*max_ptr, x[i]); }
    });
  }

  static void Exp(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, FloatingPointType* y) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) { y[i] = std::exp(x[i]); }
    });
  }

  static void Sum(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, FloatingPointType* sum_ptr) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      *sum_ptr = 0;
      for (int64_t i = 0; i < n; ++i) { *sum_ptr += x[i]; }
    });
  }

  static void Div(const KernelCtx& ctx, const int64_t n, FloatingPointType* x,
                  const FloatingPointType alpha) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) { x[i] = x[i] / alpha; }
    });
  }

  static void Mul(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, const FloatingPointType* y,
                  FloatingPointType* z) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) { z[i] = x[i] * y[i]; }
    });
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

  static void col2im(const KernelCtx& ctx, const FloatingPointType* data_col,
                     const int channels, const int height, const int width,
                     const int kernel_h, const int kernel_w, const int pad_h,
                     const int pad_w, const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w,
                     FloatingPointType* mut_dptr) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() mutable {
      memset(mut_dptr, 0, height * width * channels);
      const int output_h =
          (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h
          + 1;
      const int output_w =
          (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w
          + 1;
      const int channel_size = height * width;
      for (int channel = channels; channel--; mut_dptr += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
          for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
            int input_row = -pad_h + kernel_row * dilation_h;
            for (int output_rows = output_h; output_rows; output_rows--) {
              if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                data_col += output_w;
              } else {
                int input_col = -pad_w + kernel_col * dilation_w;
                for (int output_col = output_w; output_col; output_col--) {
                  if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                    mut_dptr[input_row * width + input_col] += *data_col;
                  }
                  data_col++;
                  input_col += stride_w;
                }
              }
              input_row += stride_h;
            }
          }
        }
      }
    });
  }

  static void im2col(const KernelCtx& ctx, const FloatingPointType* dptr,
                     const int channels, const int height, const int width,
                     const int kernel_h, const int kernel_w, const int pad_h,
                     const int pad_w, const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w,
                     FloatingPointType* data_col) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() mutable {
      const int output_h =
          (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h
          + 1;
      const int output_w =
          (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w
          + 1;
      const int channel_size = height * width;
      for (int channel = channels; channel--; dptr += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
          for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
            int input_row = -pad_h + kernel_row * dilation_h;
            for (int output_rows = output_h; output_rows; output_rows--) {
              if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                for (int output_cols = output_w; output_cols; output_cols--) {
                  *(data_col++) = 0;
                }
              } else {
                int input_col = -pad_w + kernel_col * dilation_w;
                for (int output_col = output_w; output_col; output_col--) {
                  if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                    *(data_col++) = dptr[input_row * width + input_col];
                  } else {
                    *(data_col++) = 0;
                  }
                  input_col += stride_w;
                }
              }
              input_row += stride_h;
            }
          }
        }
      }
    });
  }

  static void Fill(const KernelCtx& ctx, const FillConf& fill_conf,
                   Blob* blob) {
    if (fill_conf.has_constant_conf()) {
      ConstantFill(ctx, fill_conf.constant_conf(), blob);
    } else if (fill_conf.has_uniform_conf()) {
      UniformFill(ctx, fill_conf.uniform_conf(), blob);
    } else if (fill_conf.has_gaussian_conf()) {
      GaussianFill(ctx, fill_conf.gaussian_conf(), blob);
    } else {
      UNEXPECTED_RUN();
    }
  }

 private:
  static void ConstantFill(const KernelCtx& ctx,
                           const ConstantFillConf& fill_conf, Blob* blob) {
    FloatingPointType* dptr = static_cast<FloatingPointType*>(blob->mut_dptr());
    const int64_t elem_cnt = blob->shape().elem_cnt();
    const FloatingPointType value = fill_conf.value();
    CHECK(elem_cnt);
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = value; }
    });
  }

  static void UniformFill(const KernelCtx& ctx,
                          const UniformFillConf& fill_conf, Blob* blob) {
    CHECK(blob->shape().elem_cnt());
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      RngUniform<FloatingPointType>(
          blob->shape().elem_cnt(),
          static_cast<FloatingPointType>(fill_conf.min()),
          static_cast<FloatingPointType>(fill_conf.max()),
          static_cast<FloatingPointType*>(blob->mut_dptr()));
    });
  }

  static void GaussianFill(const KernelCtx& ctx,
                           const GaussianFillConf& fill_conf, Blob* blob) {
    CHECK(blob->shape().elem_cnt());
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      RngGaussian<FloatingPointType>(
          blob->shape().elem_cnt(),
          static_cast<FloatingPointType>(fill_conf.mean()),
          static_cast<FloatingPointType>(fill_conf.std()),
          static_cast<FloatingPointType*>(blob->mut_dptr()));
    });
  }
};

INSTANTIATE_CPU_KERNEL_UTIL_CLASS(KernelUtil);

}  //  namespace oneflow
