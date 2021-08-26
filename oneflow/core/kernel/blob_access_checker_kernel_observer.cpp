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
void ForEachObnAndIsHeaderInferedBeforeCompute(
    const Kernel* kernel, const std::function<Blob*(const std::string&)>& BnInOp2Blob,
    const HandlerT& Handler) {
  const auto& modifier_map =
      kernel->op_attribute().arg_modifier_signature().obn2output_blob_modifier();
  for (const std::string& obn : kernel->op_attribute().output_bns()) {
    Blob* blob = BnInOp2Blob(obn);
    if (blob) {
      bool is_header_infered_before_compute = modifier_map.at(obn).header_infered_before_compute();
      Handler(obn, is_header_infered_before_compute);
    }
  }
}

template<typename HandlerT>
void ForEachObnAndIsMutableByConsumer(const Kernel* kernel,
                                      const std::function<Blob*(const std::string&)>& BnInOp2Blob,
                                      const HandlerT& Handler) {
  const auto& modifier_map =
      kernel->op_attribute().arg_modifier_signature().obn2output_blob_modifier();
  for (const std::string& obn : kernel->op_attribute().output_bns()) {
    Blob* blob = BnInOp2Blob(obn);
    if (blob) {
      bool is_mutable_by_consumer = modifier_map.at(obn).is_mutable();
      Handler(obn, is_mutable_by_consumer);
    }
  }
}

void SetOutputBlobProducerInferAccessChecker(
    const Kernel* kernel, const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  ForEachObnAndIsHeaderInferedBeforeCompute(
      kernel, BnInOp2Blob, [&](const std::string& obn, bool _) {
        BnInOp2Blob(obn)->set_blob_access_checker(Global<BlobAccessCheckerIf<true, false>>::Get());
      });
}

void SetOutputBlobProducerComputeAccessChecker(
    const Kernel* kernel, const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  ForEachObnAndIsHeaderInferedBeforeCompute(
      kernel, BnInOp2Blob, [&](const std::string& obn, bool is_header_infered_before_compute) {
        const BlobAccessChecker* checker = nullptr;
        if (is_header_infered_before_compute) {
          checker = Global<BlobAccessCheckerIf<false, true>>::Get();
        } else {
          checker = Global<BlobAccessCheckerIf<true, true>>::Get();
        }
        BnInOp2Blob(obn)->set_blob_access_checker(checker);
      });
}

void SetOutputBlobConsumerAccessChecker(
    const Kernel* kernel, const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  ForEachObnAndIsMutableByConsumer(kernel, BnInOp2Blob,
                                   [&](const std::string& obn, bool is_mutable) {
                                     const BlobAccessChecker* checker = nullptr;
                                     if (is_mutable) {
                                       checker = Global<BlobAccessCheckerIf<false, true>>::Get();
                                     } else {
                                       checker = Global<BlobAccessCheckerIf<false, false>>::Get();
                                     }
                                     BnInOp2Blob(obn)->set_blob_access_checker(checker);
                                   });
}

}  // namespace

void BlobAccessCheckerKernelObserver::WillForwardHeader(
    const KernelCtx& kernel_ctx, const Kernel* kernel,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  SetOutputBlobProducerInferAccessChecker(kernel, BnInOp2Blob);
}

void BlobAccessCheckerKernelObserver::WillForwardDataContent(
    const KernelCtx& kernel_ctx, const Kernel* kernel,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  SetOutputBlobProducerComputeAccessChecker(kernel, BnInOp2Blob);
}

void BlobAccessCheckerKernelObserver::DidForwardDataContent(
    const KernelCtx& kernel_ctx, const Kernel* kernel,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  SetOutputBlobConsumerAccessChecker(kernel, BnInOp2Blob);
}

}  // namespace oneflow
