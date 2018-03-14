#include "oneflow/core/actor/embedding_lookup_model_diff_accumulate_compute_actor.h"

namespace oneflow {

void EmbeddingLookupMdDiffAccCompActor::VirtualLaunchKernel(
    Regst* in_regst, Regst* out_regst, KernelCtx kernel_ctx) {
  kernel_ctx.other = &do_cnt_;
  AsyncLaunchKernel(kernel_ctx, [this](int64_t regst_desc_id) -> Regst* {
    Regst* regst = GetCurWriteableRegst(regst_desc_id);
    if (regst == nullptr) {
      CHECK_EQ(regst_desc_id, pending_in_regst_.front()->regst_desc_id());
      return pending_in_regst_.front();
    } else {
      return regst;
    }
  });
  do_cnt_ += 1;
}

REGISTER_ACTOR(TaskType::kEmbeddingLookupMdDiffAcc,
               EmbeddingLookupMdDiffAccCompActor);

}  // namespace oneflow
