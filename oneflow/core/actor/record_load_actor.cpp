#include "oneflow/core/actor/record_load_actor.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

void RecordLoadActor::VirtualCompActorInit(const TaskProto& task_proto) {
  is_eof_ = false;
  OF_SET_MSG_HANDLER(&RecordLoadActor::HandlerNormal);
  record_load_status_.is_eof = false;
  record_load_status_.record_num = 0;
}

void RecordLoadActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &record_load_status_;
  AsyncLaunchKernel(kernel_ctx);
  CHECK_EQ(record_load_status_.is_eof, false);
}

void RecordLoadActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  Regst* in_regst = GetNaiveCurReadable("in");
  HandleProducedNaiveDataRegstToConsumer([&](Regst* out_regst) {
    out_regst->set_piece_id(in_regst->piece_id());
    return true;
  });
}

REGISTER_ACTOR(kRecordLoad, RecordLoadActor);

}  // namespace oneflow
