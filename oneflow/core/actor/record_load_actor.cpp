#include "oneflow/core/actor/record_load_actor.h"
#include "oneflow/core/persistence/cyclic_persistent_in_stream.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/operator/record.pb.h"

namespace oneflow {

void RecordLoadActor::VirtualCompActorInit(const TaskProto& task_proto) {
  record_type = task_proto.record_type();
  piece_id_ = 0;
  is_eof_ = false;
  OF_SET_MSG_HANDLER(&RecordLoadActor::HandlerWaitToStart);
  if (JobDesc::Singleton()->IsTrain()) {
    in_stream_.reset(
        new CyclicPersistentInStream(GlobalFS(), task_proto.data_path()));
  } else {
    in_stream_.reset(
        new NormalPersistentInStream(GlobalFS(), task_proto.data_path()));
  }
}

int RecordLoadActor::HandlerWaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  ActUntilFail();
  OF_SET_MSG_HANDLER(&RecordLoadActor::HandlerNormal);
  return 0;
}

int RecordLoadActor::HandlerNormal(const ActorMsg& msg) {
  CHECK_EQ(msg.msg_type(), ActorMsgType::kRegstMsg);
  TryUpdtStateAsProducedRegst(msg.regst());
  ActUntilFail();
  return TrySwitchToZombieOrFinish();
}

void RecordLoadActor::Act() {
  Regst* regst = GetCurSoleWriteableRegst();
  regst->set_piece_id(piece_id_++);
  size_t record_size = 0;
  size_t i = 0;
  for (; i < JobDesc::Singleton()->SinglePieceSize(); ++i) {
    if (!in_stream_->Read(reinterpret_cast<char*>(&record_size),
                          sizeof(size_t))) {
      std::unique_ptr<std::vector<char>> buffer =
          of_make_unique<std::vector<char>>(record_size);
      CHECK_EQ(in_stream_->Read(buffer->data(), record_size), 0);
      regst->GetRecordBlob<OFRecord>()->mut_records(i)->ParseFromArray(
          buffer->data(), record_size);
    } else {
      is_eof_ = true;
      break;
    }
  }
  if (i != 0) {
    regst->GetRecordBlob<OFRecord>()->set_record_num(i);
    AsyncSendRegstMsgToConsumer();
  }
}

bool RecordLoadActor::IsReadAlwaysUnReadyFromNow() {
  return is_eof_ || piece_id_ >= RuntimeCtx::Singleton()->total_piece_num();
}

REGISTER_ACTOR(kRecordLoad, RecordLoadActor);

}  // namespace oneflow
