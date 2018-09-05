#include "oneflow/core/actor/reduce_add2_compute_actor.h"

namespace oneflow {

void ReduceAdd2CompActor::VirtualCompActorInit(const TaskProto& proto) {
  InputWiseCompActor::Init(proto);
  int32_t out_num = proto.exec_sequence()
                        .exec_node()
                        .Get(0)
                        .kernel_conf()
                        .op_attribute()
                        .op_conf()
                        .reduce_local_add_conf()
                        .out_num();
  int64_t parallel_num = parallel_ctx()->parallel_num();
  int64_t parallel_id = parallel_ctx()->parallel_id();
  int64_t dev_num_of_each_machine = parallel_num / out_num;

  out_blob_init_status_.resize(out_num, false);
  int64_t inplace_in_blob_id = parallel_id % dev_num_of_each_machine;
  inplace_blob_ids_.emplace(inplace_in_blob_id);
  for (int64_t i = 0; i < out_num - 1; ++i) {
    inplace_in_blob_id += dev_num_of_each_machine;
    inplace_blob_ids_.emplace(inplace_in_blob_id);
  }
  CHECK_LT(inplace_in_blob_id, parallel_num);
}

void ReduceAdd2CompActor::SetKernelCtxOther(void** other) {
  int64_t dev_num_of_each_machine = parallel_ctx()->parallel_num() / out_blob_init_status_.size();
  int64_t in_bn_id = InBnId4RegstDescId(cur_processed_regst_desc_id());
  cur_out_blob_id_ = in_bn_id / dev_num_of_each_machine;
  bool is_inited = EnableInplace() ? true : out_blob_init_status_.at(cur_out_blob_id_);
  bool is_inplace_in_bn_id =
      EnableInplace() ? inplace_blob_ids_.find(in_bn_id) != inplace_blob_ids_.end() : false;

  other_val_ = std::make_tuple(in_bn_id, cur_out_blob_id_, is_inited, is_inplace_in_bn_id);
  *other = static_cast<void*>(&other_val_);
}

void ReduceAdd2CompActor::VirtualUpdateMemberStatusAfterAct() {
  if (out_blob_init_status_.at(cur_out_blob_id_) == false) {
    out_blob_init_status_.at(cur_out_blob_id_) = true;
  }
}

void ReduceAdd2CompActor::VirtualUpdateMemberStatusAfterSendRegstMsgToConsumer() {
  for (int32_t i = 0; i < out_blob_init_status_.size(); ++i) {
    CHECK(out_blob_init_status_.at(i));
    out_blob_init_status_.at(i) = false;
  }
}

REGISTER_ACTOR(TaskType::kReduceAdd2, ReduceAdd2CompActor);

}  // namespace oneflow
