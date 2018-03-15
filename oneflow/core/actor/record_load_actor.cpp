#include "oneflow/core/actor/record_load_actor.h"
#include "oneflow/core/persistence/cyclic_persistent_in_stream.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/operator/record.pb.h"

namespace oneflow {

static const int32_t record_load_regst_num = 2;

void RecordLoadActor::Init(const TaskProto& task_proto, const ThreadCtx&) {
  set_actor_id(task_proto.task_id());
  consumers_actor_id_ = PbRf2StdVec(task_proto.related_decode_task_ids());
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

int RecordLoadActor::HandlerZombie(const ActorMsg& msg) {
  CHECK_EQ(msg.msg_type(), ActorMsgType::kRegstMsg);
  TryUpdtStateAsProducedRegst(msg.regst());
  if (produced_regsts_.size() == 0) {
    set_msg_handler(static_cast<MsgHandler>(nullptr));
    return 1;
  }
  return 0;
}

int RecordLoadActor::TrySwitchToZombieOrFinish() {
  if (IsLoadDone()) {
    for (int64_t consumer : consumers_actor_id_) {
      ActorMsg msg = ActorMsg::BuildEordMsg(consumer, -1);
      ActorMsgBus::Singleton()->SendMsg(std::move(msg));
    }
    if (produced_regsts_.size() == 0) {
      OF_SET_MSG_HANDLER(nullptr);
      return 1;
    } else {
      OF_SET_MSG_HANDLER(&RecordLoadActor::HandlerZombie);
      return 0;
    }
  }
  return 0;
}

void RecordLoadActor::TryUpdtStateAsProducedRegst(Regst* regst) {
  auto reading_cnt_it = produced_regst2reading_cnt_.find(regst);
  CHECK(reading_cnt_it != produced_regst2reading_cnt_.end());
  CHECK_GE(reading_cnt_it->second, 1);
  reading_cnt_it->second -= 1;
  if (reading_cnt_it->second == 0) {
    CHECK_EQ(regst, produced_regsts_.front().get());
    produced_regsts_.pop();
    produced_regst2reading_cnt_.erase(reading_cnt_it);
  }
}

void RecordLoadActor::ActUntilFail() {
  while (produced_regsts_.size() < record_load_regst_num && !IsLoadDone()) {
    Act();
  }
}

void RecordLoadActor::Act() {
  std::unique_ptr<Regst> regst(RegstMgr::Singleton()->NewRegst());
  regst->set_piece_id(piece_id_++);
  size_t record_size = 0;
  size_t i = 0;
  for (; i < JobDesc::Singleton()->SinglePieceSize(); ++i) {
    if (!in_stream_->Read(reinterpret_cast<char*>(&record_size),
                          sizeof(size_t))) {
      std::unique_ptr<std::vector<char>> buffer =
          of_make_unique<std::vector<char>>(record_size);
      CHECK_EQ(in_stream_->Read(buffer->data(), record_size), 0);
      regst->GetRecordBlob<OfRecord>()->mut_records(i)->ParseFromArray(
          buffer->data(), record_size);
    } else {
      is_eof_ = true;
      break;
    }
  }
  if (i != 0) {
    CHECK(produced_regst2reading_cnt_
              .emplace(regst.get(), consumers_actor_id_.size())
              .second);
    for (int64_t consumer : consumers_actor_id_) {
      ActorMsg msg =
          ActorMsg::BuildRegstMsgToConsumer(actor_id(), consumer, regst.get());
      ActorMsgBus::Singleton()->SendMsg(std::move(msg));
    }
    produced_regsts_.push(std::move(regst));
  }
}

bool RecordLoadActor::IsLoadDone() {
  return is_eof_ || piece_id_ >= RuntimeCtx::Singleton()->total_piece_num();
}

REGISTER_ACTOR(kRecordLoad, RecordLoadActor);

}  // namespace oneflow
