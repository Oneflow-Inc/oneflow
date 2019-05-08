#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/blas.h"

namespace oneflow {

namespace {

template<typename T>
static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
  const int m, const int n, const int k, const T alpha, const T* a, const T* b,
            const T beta, T* c) {
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

  NewKernelUtil<kGPU>::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a->dptr<T>(), b->dptr<T>(), beta,
           c->mut_dptr<T>());
}

template<typename T>
static void Addition1DImpl(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0) {
  for (int64_t i = 0; i != n; ++i) { out[i] = in_0[i]; }
}

template<typename T>
static void Addition2DImpl(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1) {
  for (int64_t i = 0; i != n; ++i) { out[i] = in_0[i] + in_1[i]; }
}

template<typename T>
static void Addition3DImpl(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2) {
  for (int64_t i = 0; i != n; ++i) { out[i] = in_0[i] + in_1[i] + in_2[i]; }
}

template<typename T>
static void Addition4DImpl(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3) {
  for (int64_t i = 0; i != n; ++i) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i]; }
}

template<typename T>
static void Addition5DImpl(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4) {
  for (int64_t i = 0; i != n; ++i) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i]; }
}

template<typename T>
static void Addition6DImpl(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4, const T* in_5) {
   for (int64_t i = 0; i != n; ++i) {
    out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i];
  }
}

template<typename T>
static void Addition7DImpl(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4, const T* in_5, const T* in_6) {
  for (int64_t i = 0; i != n; ++i) {
    out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i];
  }
}

template<typename T>
static void Addition8DImpl(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4, const T* in_5, const T* in_6,
                       const T* in_7) {
  for (int64_t i = 0; i != n; ++i) {
    out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i] + in_7[i];
  }
}

template<typename T>
static void Addition9DImpl(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4, const T* in_5, const T* in_6,
                       const T* in_7, const T* in_8) {
  for (int64_t i = 0; i != n; ++i) {
    out[i] =
        in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i] + in_7[i] + in_8[i];
}

} // namespace

#define CPU_KU_METHOD void NewKernelUtil<DeviceType::kCPU>::

CPU_KU_METHOD BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                      float alpha, float beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}

CPU_KU_METHOD BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       double alpha, double beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}

CPU_KU_METHOD BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                      float16 alpha, float16 beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}

CPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const float alpha, const float* a, const float* b,
            const float beta, float* c) {
  Gemm(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

CPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const double alpha, const double* a, const double* b,
            const double beta, double* c) {
  Gemm(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

CPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const float16 alpha, const float16* a, const float16* b,
            const float16 beta, float16* c) {
   UNIMPLEMENTED();
}

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0) {
  Addition1DImpl<float>(ctx, n, out, in_0);
}

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0) {
  Addition1DImpl<double>(ctx, n, out, in_0);
}

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1) {
  Addition2DImpl<float>(ctx, n, out, in_0, int_1);
};

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1) {
  Addition2DImpl<double>(ctx, n, out, in_0, in_1);
};

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1,
          const float* in_2) {
  Addition3DImpl<float>(ctx, n, out, in_0, in_1, in_2);
};

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1,
          const double* in_2) {
  Addition3DImpl<double>(ctx, n, out, in_0, in_1, in_2);
};

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1,
          const float* in_2, const float* in_3) {
  Addition4DImpl<float>l(ctx, n, out, in_0, in_1, in_2, in_3);
};

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1,
          const double* in_2, const double* in_3) {
  Addition4DImpl<double>(ctx, n, out, in_0, in_1, in_2, in_3);
};

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1,
          const float* in_2, const float* in_3, const float* in_4) {
  Addition5DImpl<float>(ctx, n, out, in_0, in_1, in_2, in_3, in_4);
 };

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1,
          const double* in_2, const double* in_3, const double* in_4) {
  Addition5DImpl<double>(ctx, n, out, in_0, in_1, in_2, in_3, in_4);
};
CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1,
          const float* in_2, const float* in_3, const float* in_4, const float* in_5) {
  Addition6DImpl<float>(ctx, n, out, in_0, in_1, in_2, in_3, in_4, in_5);
};
CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1,
          const double* in_2, const double* in_3, const double* in_4, const double* in_5) {
  Addition6DImpl<double>(ctx, n, out, in_0, in_1, in_2, in_3, in_4, in_5);
};

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1,
          const float* in_2, const float* in_3, const float* in_4, const float* in_5, const float* in_6) {
  Addition7DImpl<float>(ctx, n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6);
};

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1,
          const double* in_2, const double* in_3, const double* in_4, const double* in_5, const double* in_6) {
  Addition7DImpl<double>(ctx, n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6);
};

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1,
          const float* in_2, const float* in_3, const float* in_4, const float* in_5, const float* in_6,
          const float* in_7) {
  Addition8DImpl<float>(ctx, n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7);
};

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1,
          const double* in_2, const double* in_3, const double* in_4, const double* in_5, const double* in_6,
          const double* in_7) {
  Addition8DImpl<double>(ctx, n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7);
};

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1,
          const float* in_2, const float* in_3, const float* in_4, const float* in_5, const float* in_6,
          const float* in_7, const float* in_8) {
  Addition9DImpl<float>(ctx, n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8);
};

CPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1,
          const double* in_2, const double* in_3, const double* in_4, const double* in_5, const double* in_6,
          const double* in_7, const double* in_8) {
  Addition9DImpl<double>(ctx, n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8);
};

} // namespace oneflow

