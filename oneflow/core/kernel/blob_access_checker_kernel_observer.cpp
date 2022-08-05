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
#include "oneflow/core/kernel/blob_access_checker_kernel_observer.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename HandlerT>
void ForEachObnAndIsHeaderInferedBeforeCompute(KernelContext* kernel_ctx, const Kernel* kernel,
                                               const HandlerT& Handler) {
  const auto& modifier_map =
      kernel->op_attribute().arg_modifier_signature().obn2output_blob_modifier();
  for (const std::string& obn : kernel->op_attribute().output_bns()) {
    Blob* blob = kernel_ctx->BnInOp2Blob(obn);
    if (blob) {
      bool is_header_infered_before_compute = modifier_map.at(obn).header_infered_before_compute();
      Handler(obn, is_header_infered_before_compute);
    }
  }
}

template<typename HandlerT>
void ForEachObnAndIsMutableByConsumer(KernelContext* kernel_ctx, const Kernel* kernel,
                                      const HandlerT& Handler) {
  const auto& modifier_map =
      kernel->op_attribute().arg_modifier_signature().obn2output_blob_modifier();
  for (const std::string& obn : kernel->op_attribute().output_bns()) {
    Blob* blob = kernel_ctx->BnInOp2Blob(obn);
    if (blob) {
      bool is_mutable_by_consumer = modifier_map.at(obn).is_mutable();
      Handler(obn, is_mutable_by_consumer);
    }
  }
}

void SetOutputBlobProducerInferAccessChecker(KernelContext* kernel_ctx, const Kernel* kernel) {
  ForEachObnAndIsHeaderInferedBeforeCompute(
      kernel_ctx, kernel, [&](const std::string& obn, bool _) {
        kernel_ctx->BnInOp2Blob(obn)->set_blob_access_checker(
            Singleton<BlobAccessCheckerIf<true, false>>::Get());
      });
}

void SetOutputBlobProducerComputeAccessChecker(KernelContext* kernel_ctx, const Kernel* kernel) {
  ForEachObnAndIsHeaderInferedBeforeCompute(
      kernel_ctx, kernel, [&](const std::string& obn, bool is_header_infered_before_compute) {
        const BlobAccessChecker* checker = nullptr;
        if (is_header_infered_before_compute) {
          checker = Singleton<BlobAccessCheckerIf<false, true>>::Get();
        } else {
          checker = Singleton<BlobAccessCheckerIf<true, true>>::Get();
        }
        kernel_ctx->BnInOp2Blob(obn)->set_blob_access_checker(checker);
      });
}

void SetOutputBlobConsumerAccessChecker(KernelContext* kernel_ctx, const Kernel* kernel) {
  ForEachObnAndIsMutableByConsumer(
      kernel_ctx, kernel, [&](const std::string& obn, bool is_mutable) {
        const BlobAccessChecker* checker = nullptr;
        if (is_mutable) {
          checker = Singleton<BlobAccessCheckerIf<false, true>>::Get();
        } else {
          checker = Singleton<BlobAccessCheckerIf<false, false>>::Get();
        }
        kernel_ctx->BnInOp2Blob(obn)->set_blob_access_checker(checker);
      });
}

}  // namespace

void BlobAccessCheckerKernelObserver::WillForwardHeader(KernelContext* kernel_ctx,
                                                        const Kernel* kernel) {
  SetOutputBlobProducerInferAccessChecker(kernel_ctx, kernel);
}

void BlobAccessCheckerKernelObserver::WillForwardDataContent(KernelContext* kernel_ctx,
                                                             const Kernel* kernel) {
  SetOutputBlobProducerComputeAccessChecker(kernel_ctx, kernel);
}

void BlobAccessCheckerKernelObserver::DidForwardDataContent(KernelContext* kernel_ctx,
                                                            const Kernel* kernel) {
  SetOutputBlobConsumerAccessChecker(kernel_ctx, kernel);
}

}  // namespace oneflow
