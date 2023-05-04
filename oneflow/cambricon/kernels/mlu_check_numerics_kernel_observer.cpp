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
#include "oneflow/cambricon/kernels/mlu_check_numerics_kernel_observer.h"

#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/bang/bang_kernels.h"
#include "oneflow/cambricon/common/mlu_guard.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

static bool HasNotFiniteMlu(ep::MluStream* stream, const Blob* blob,
                            int64_t* count_not_finite_device) {
  if (blob->shape().elem_cnt() == 0) { return false; }
  if (blob->data_type() != DataType::kFloat) { return false; }
  std::vector<const float*> blob_data = {blob->dptr<float>()};
  std::vector<int64_t> sizes = {blob->shape().elem_cnt()};

  BangHandle handle(stream->mlu_stream(), stream->device()->nclusters(),
                    stream->device()->ncores_per_cluster());

  bang_multi_count_not_finite_kernel<float>(handle, 1, blob_data.data(), sizes.data(),
                                            count_not_finite_device, count_not_finite_device + 1,
                                            sizeof(int64_t));

  int64_t has_not_finite_host = 0;
  OF_MLU_CHECK(cnrtMemcpyAsync(&has_not_finite_host, count_not_finite_device, sizeof(int64_t),
                               stream->mlu_stream(), cnrtMemcpyDevToHost));
  stream->Sync();
  return has_not_finite_host > 0;
}

MluCheckNumericsKernelObserver::MluCheckNumericsKernelObserver()
    : count_not_finite_host_(nullptr), count_not_finite_device_(nullptr) {
  OF_MLU_CHECK(cnrtMalloc((void**)(&count_not_finite_device_), sizeof(int64_t) * 2));
}

MluCheckNumericsKernelObserver::~MluCheckNumericsKernelObserver() {
  OF_MLU_CHECK(cnrtFree(count_not_finite_device_));
}

void MluCheckNumericsKernelObserver::DidForwardDataContent(KernelContext* ctx,
                                                           const Kernel* kernel) {
  for (const auto& obn : kernel->op_attribute().output_bns()) {
    Blob* blob = ctx->BnInOp2Blob(obn);
    if (blob != nullptr) {
      bool has_not_finite =
          HasNotFiniteMlu(ctx->stream()->As<ep::MluStream>(), blob, count_not_finite_device_);
      CHECK(!has_not_finite) << kernel->op_conf().name() << " : " << obn << " has nan or inf";
    }
  }
}

}  // namespace oneflow
