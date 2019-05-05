#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

void NewKernelUtil::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const float alpha, const float* a, const float* b,
            const float beta, float* c) {

}
void NewKernelUtil::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const double alpha, const double* a, const double* b,
            const double beta, double* c) {

}
void NewKernelUtil::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
            const int m, const int n, const int k, const float16 alpha, const float16* a, const float16* b,
            const float16 beta, float16* c) {

}

} // namespace oneflow
