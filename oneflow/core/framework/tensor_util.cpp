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
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/common/blocking_then_busy.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

Maybe<void> SyncAccessTensorWithTimeOut(
    const std::shared_ptr<Tensor>& tensor,
    const std::function<void(ep::Stream*, const std::shared_ptr<vm::EagerBlobObject>&)>& Callback,
    const std::string& modifier) {
  auto btb = std::make_shared<BlockingThenBusy>(1);
  auto local_tensor = JUST(tensor->AsLocalTensor());
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->SyncAccessBlobByCallback(local_tensor, btb, Callback, modifier);
  }));
  JUST(btb->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  return Maybe<void>::Ok();
}

Maybe<void> CopyLocalTensorDataTo(const std::shared_ptr<Tensor>& input, void* mem_ptr,
                                  size_t size) {
  CHECK_OR_RETURN(input->is_local());  // NOLINT
  CHECK_OR_RETURN(input->is_contiguous()) << Error::RuntimeError() << kOfBugIssueUploadPrompt;
  CHECK_EQ_OR_RETURN(input->shape()->elem_cnt() * JUST(input->dtype()->bytes()), size)
      << Error::RuntimeError() << kOfBugIssueUploadPrompt;
  std::shared_ptr<one::LocalTensor> local_tensor = JUST(input->AsLocalTensor());
  const auto& Callback = [&](ep::Stream* stream,
                             const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
    SyncAutoMemcpy(stream, mem_ptr, eager_blob_object->dptr(), size, memory::MakeHostMemCase(),
                   eager_blob_object->mem_case());
  };
  auto btb = std::make_shared<BlockingThenBusy>(1);
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->SyncAccessBlobByCallback(local_tensor, btb, Callback, "const");
  }));
  JUST(btb->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  return Maybe<void>::Ok();
}

Maybe<Scope> GetTensorScope(const std::shared_ptr<Tensor>& tensor) {
  CHECK_OR_RETURN(LazyMode::is_enabled())
      << "it's not allowed to access tensor scope in eager mode";
  const auto& lbn = TensorNameScope::Global()->Lookup(tensor);
  CHECK_OR_RETURN(!lbn.empty()) << "can not access tensor scope since it is not a lazy tensor or a "
                                   "captured eager tensor in graph";
  const auto& infer_ctx = JUST(GetCurInferCtx());
  auto lbi = GenLogicalBlobId(lbn);
  const auto* op = JUST(infer_ctx->Op4OpName(lbi.op_name()));
  return Singleton<symbol::Storage<Scope>>::Get()->MaybeGetPtr(op->op_conf().scope_symbol_id());
}

template<typename T>
Maybe<T> GetItemInScalarTensor(const std::shared_ptr<Tensor>& scalar_tensor) {
  CHECK_OR_RETURN(scalar_tensor->is_eager()) << "Only eager scalar tensor support GetItem";
  std::shared_ptr<LocalTensor> local_tensor;
  {
    auto tensor = scalar_tensor;
    if (tensor->is_global()) {
      Symbol<ParallelDesc> parallel_desc;
      {
        const ParallelConf parallel_conf = GenParallelConfOfCpuOnAllRanks();
        JUST(PhysicalRun(
            [&parallel_desc, &parallel_conf](InstructionsBuilder* builder) -> Maybe<void> {
              parallel_desc = SymbolOf(*JUST(builder->GetParallelDescSymbol(parallel_conf)));
              return Maybe<void>::Ok();
            }));
      }
      const auto& broadcast_sbp = JUST(MakeBroadcastSbpParallel());
      tensor = JUST(functional::ToGlobal(tensor, parallel_desc, {broadcast_sbp}, /*grad_sbp=*/{},
                                         /*check_meta=*/false, /*copy=*/false));
      tensor = JUST(functional::GlobalToLocal(tensor, /*copy=*/false));
    }
    local_tensor = JUST(tensor->AsLocalTensor());
  }

  T scalar = 0;
  {
    const auto& Callback = [&](ep::Stream* stream,
                               const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
      SyncAutoMemcpy(stream, &scalar, eager_blob_object->mut_dptr(), sizeof(T),
                     memory::MakeHostMemCase(), eager_blob_object->mem_case());
    };
    auto btb = std::make_shared<BlockingThenBusy>(1);
    JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      return builder->SyncAccessBlobByCallback(local_tensor, btb, Callback, "const");
    }));
    JUST(btb->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  }
  return scalar;
}

#define SPECIALIZE_SCALAR_TENSOR_TO_SCALAR_FUNC(cpp_type, of_type) \
  template Maybe<cpp_type> GetItemInScalarTensor(const std::shared_ptr<Tensor>& scalar_tensor);

OF_PP_FOR_EACH_TUPLE(SPECIALIZE_SCALAR_TENSOR_TO_SCALAR_FUNC, POD_DATA_TYPE_SEQ);
#undef SPECIALIZE_SCALAR_TENSOR_TO_SCALAR_FUNC
}  // namespace one
}  // namespace oneflow
