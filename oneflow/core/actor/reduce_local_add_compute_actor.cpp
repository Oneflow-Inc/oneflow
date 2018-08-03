#include "oneflow/core/actor/reduce_local_add_compute_actor.h"

namespace oneflow {

void ReduceLocalAddCompActor::VirtualCompActorInit(const TaskProto& proto) {
  InputWiseCompActor::Init(proto);
  int32_t out_num = proto.exec_sequence()
                        .exec_node()
                        .Get(0)
                        .kernel_conf()
                        .op_attribute()
                        .op_conf()
                        .reduce_local_add_conf()
                        .out_num();
  out_blob_init_status_.resize(out_num, false);
}

void ReduceLocalAddCompActor::SetKernelCtxOther(void** other) {
  int64_t in_bn_id = InBnId4RegstDescId(cur_processed_regst_desc_id());
  int64_t dev_num_of_each_machine = parallel_ctx()->parallel_num() / out_blob_init_status_.size();
  cur_out_blob_id_ = in_bn_id / dev_num_of_each_machine;
  other_val_ =
      std::make_tuple(in_bn_id, cur_out_blob_id_, out_blob_init_status_.at(cur_out_blob_id_));
  *other = static_cast<void*>(&other_val_);
}

void ReduceLocalAddCompActor::VirtualUpdateMemberStatusAfterAct() {
  if (out_blob_init_status_.at(cur_out_blob_id_) == false) {
    out_blob_init_status_.at(cur_out_blob_id_) = true;
  }
}

void ReduceLocalAddCompActor::VirtualUpdateMemberStatusAfterSendRegstMsgToConsumer() {
  for (int32_t i = 0; i < out_blob_init_status_.size(); ++i) {
    CHECK(out_blob_init_status_.at(i));
    out_blob_init_status_.at(i) = false;
  }
}

REGISTER_ACTOR(TaskType::kReduceLocalAdd, ReduceLocalAddCompActor);

}  // namespace oneflow
