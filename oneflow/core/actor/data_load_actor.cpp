#include "oneflow/core/actor/data_load_actor.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

void DataLoadActor::VirtualCompActorInit(const TaskProto& task_proto) {
  piece_id_ = 0;
  is_eof_ = false;
  OF_SET_MSG_HANDLER(&DataLoadActor::HandlerWaitToStart);
  data_load_status_.is_eof = false;
  // record_load_status_.record_num = 0;
}

void DataLoadActor::Act() {
  Regst* regst = GetNaiveCurWriteable("record");
  regst->set_piece_id(piece_id_++);

  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &data_load_status_;
  AsyncLaunchKernel(kernel_ctx);
  if (data_load_status_.is_eof) { is_eof_ = true; }
}

// void DataLoadActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
//   if (record_load_status_.record_num > 0) { HandleProducedNaiveDataRegstToConsumer(); }
// }

bool DataLoadActor::IsCustomizedReadReady() {
  return !is_eof_ && piece_id_ < Global<RuntimeCtx>::Get()->total_piece_num();
}

int DataLoadActor::HandlerWaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  OF_SET_MSG_HANDLER(&DataLoadActor::HandlerNormal);
  return ProcessMsg(msg);
}

REGISTER_ACTOR(kDataLoad, DataLoadActor);

}  // namespace oneflow
