#include <cub/cub.cuh>
#include <math.h>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

cublasOperation_t CblasTrans2CublasTrans(CBLAS_TRANSPOSE trans) {
  cublasOperation_t cublas_trans;
  if (trans == CBLAS_TRANSPOSE::CblasNoTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_N;
  } else if (trans == CBLAS_TRANSPOSE::CblasTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_T;
  } else if (trans == CBLAS_TRANSPOSE::CblasConjTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_C;
  } else {
    // do nothing
  }
  return cublas_trans;
}

template<typename T>
static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
  const int m, const int n, const int k, const T alpha, const T* a, const T* b,
  const T beta, T* c) {  
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;
  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);

  cublas_gemm<T>(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k, &alpha, b,
  ldb, a, lda, &beta, c, ldc);
}

static void HGemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
  const int m, const int n, const int k, const float16 alpha, const float16* a, const float16* b,
  const float16 beta, float16* c) {
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;

  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
  CudaCheck(cublasHgemm(ctx->cublas_tensor_op_math_handle(), cublas_trans_b, cublas_trans_a, n, m,
        k, reinterpret_cast<const half*>(&alpha),
        reinterpret_cast<const half*>(b), ldb, reinterpret_cast<const half*>(a),
        lda, reinterpret_cast<const half*>(&beta), reinterpret_cast<half*>(c),
        ldc));
}

template<typename T>
static void BlobGemmImpl(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                      T alpha, T beta, const Blob* a, const Blob* b, Blob* c) {
  const int m = c->shape().At(0);
  const int n = c->shape().Count(1);
  const int k = (trans_a == CblasNoTrans) ? a->shape().Count(1) : a->shape().At(0);

  NewKernelUtil<DeviceType::kGPU>::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a->dptr<T>(), b->dptr<T>(), beta,
           c->mut_dptr<T>());
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i]; }
}
template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i] + in_2[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4, const T* in_5) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4, const T* in_5, const T* in_6) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i];
  }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4, const T* in_5, const T* in_6, const T* in_7) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i] + in_7[i];
  }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4, const T* in_5, const T* in_6, const T* in_7,
                        const T* in_8) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[i] =
        in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i] + in_7[i] + in_8[i];
  }
}

template<typename T>
__global__ void gpu_add() {
  
}

} // namespace

#define GPU_KU_METHOD void NewKernelUtil<DeviceType::kGPU>::

GPU_KU_METHOD BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
  float alpha, float beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}
GPU_KU_METHOD BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
  double alpha, double beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}
GPU_KU_METHOD BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
  float16 alpha, float16 beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}
GPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
    const int m, const int n, const int k, const float alpha, const float* a, const float* b,
const float beta, float* c) {
  Gemm<float>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}
GPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
    const int m, const int n, const int k, const double alpha, const double* a, const double* b,
    const double beta, double* c) {
  Gemm<double>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}
GPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
    const int m, const int n, const int k, const float16 alpha, const float16* a, const float16* b,
    const float16 beta, float16* c) {
  HGemm(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0) {
  gpu_add<float>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0);
}

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0) {
  gpu_add<double>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0);
}

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float16* out, const float16* in_0) {
  gpu_add<float16>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0);
}

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1) {
  gpu_add<float>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1);
};

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1) {
  gpu_add<double>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1);
};

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float16* out, const float16* in_0, const float16* in_1) {
  gpu_add<float16>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, reinterpret_cast<half*>(out), reinterpret_cast<half*>(in_0), reinterpret_cast<half*>(in_1));
};

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1,
          const float* in_2) {
  gpu_add<float>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1, in_2);
};

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1,
          const double* in_2) {
  gpu_add<double>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1, in_2);
};

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1,
          const float* in_2, const float* in_3) {
  gpu_add<float>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1, in_2, in_3);
};

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1,
          const double* in_2, const double* in_3) {
  gpu_add<double>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1, in_2, in_3);
};

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1,
          const float* in_2, const float* in_3, const float* in_4) {
  gpu_add<float>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1, in_2, in_3, in_4);
 };

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1,
          const double* in_2, const double* in_3, const double* in_4) {
  gpu_add<double>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1, in_2, in_3, in_4);
};
GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1,
          const float* in_2, const float* in_3, const float* in_4, const float* in_5) {
  gpu_add<float>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1, in_2, in_3, in_4, in_5);
};
GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1,
          const double* in_2, const double* in_3, const double* in_4, const double* in_5) {
  gpu_add<double>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1, in_2, in_3, in_4, in_5);
};

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1,
          const float* in_2, const float* in_3, const float* in_4, const float* in_5, const float* in_6) {
gpu_add<float>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6);
};

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1,
          const double* in_2, const double* in_3, const double* in_4, const double* in_5, const double* in_6) {
  gpu_add<double>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6);
};

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1,
          const float* in_2, const float* in_3, const float* in_4, const float* in_5, const float* in_6,
          const float* in_7) {
  gpu_add<float>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7);
}

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1,
          const double* in_2, const double* in_3, const double* in_4, const double* in_5, const double* in_6,
          const double* in_7) {
  gpu_add<double>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7);
};

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, float* out, const float* in_0, const float* in_1,
          const float* in_2, const float* in_3, const float* in_4, const float* in_5, const float* in_6,
          const float* in_7, const float* in_8) {
  gpu_add<float>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8);
};

GPU_KU_METHOD Addition(DeviceCtx* ctx, const int64_t n, double* out, const double* in_0, const double* in_1,
          const double* in_2, const double* in_3, const double* in_4, const double* in_5, const double* in_6,
          const double* in_7, const double* in_8) {
  gpu_add<double>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8);
};


} // namespace oneflow
