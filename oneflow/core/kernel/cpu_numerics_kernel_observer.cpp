/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/kernel/cpu_check_numerics_kernel_observer.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T>
bool HasNotFinite(const int64_t elem_cnt, const T* data_ptr) {
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    if (!std::isfinite(data_ptr[i])) { return true; }
  }
  return false;
}

bool HasNotFiniteCpu(ep::Stream* stream, const Blob* blob) {
  const DataType dtype = blob->data_type();
  const int64_t elem_cnt = blob->shape().elem_cnt();
  if (dtype == kFloat) {
    return HasNotFinite<float>(elem_cnt, blob->dptr<float>());
  } else if (dtype == kDouble) {
    return HasNotFinite<double>(elem_cnt, blob->dptr<double>());
  } else {
    return false;
  }
}

void DumpBlob(KernelContext* ctx, const std::string& bn) {
  Blob* blob = ctx->BnInOp2Blob(bn);
  if (blob != nullptr) {
    std::ofstream ofs(bn);
    ofs.write(blob->dptr<char>(), blob->ByteSizeOfBlobBody());
  }
}

void DumpBlobs(KernelContext* ctx, const Kernel* kernel) {
  for (const auto& obn : kernel->op_attribute().output_bns()) { DumpBlob(ctx, obn); }
  for (const auto& ibn : kernel->op_attribute().input_bns()) { DumpBlob(ctx, ibn); }
}

}  // namespace

void CpuCheckNumericsKernelObserver::DidForwardDataContent(KernelContext* ctx,
                                                           const Kernel* kernel) {
  for (const auto& obn : kernel->op_attribute().output_bns()) {
    Blob* blob = ctx->BnInOp2Blob(obn);
    if (blob != nullptr) {
      bool has_not_finite = HasNotFiniteCpu(ctx->stream(), blob);
      if (has_not_finite
          && ParseBooleanFromEnv("ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS_DUMP", false)) {
        DumpBlobs(ctx, kernel);
      }
      CHECK(!has_not_finite) << kernel->op_conf().name() << " : " << obn << " has nan or inf";
    }
  }
}

}  // namespace oneflow
