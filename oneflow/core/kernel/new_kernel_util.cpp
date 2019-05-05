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
            const int m, const int n, const int k, const half alpha, const half* a, const half* b,
            const half beta, half* c) {

}

} // namespace oneflow
