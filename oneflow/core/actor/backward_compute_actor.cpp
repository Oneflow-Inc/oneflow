#include "oneflow/core/actor/backward_compute_actor.h"

namespace oneflow {

void BackwardCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  activation_regst_desc_id_ = RegstDescId4Name("activation");
  data_tmp_regst_desc_id_ = RegstDescId4Name("data_tmp");
  out_regst_desc_id_ = RegstDescId4Name("out");
  in_regst_desc_id_ = RegstDescId4Name("in");
  out_diff_regst_desc_id_ = RegstDescId4Name("out_diff");
  is_out_diff_eord_ = false;
  VirtualBackwardCompActorInit(task_proto);
}

void BackwardCompActor::AsyncReturnModelRegstUntilMatchCurOutRegst(
    int64_t cur_model_id, std::queue<Regst*>& model_rq) {
  while (!model_rq.empty()
         && model_rq.front()->model_version_id() < cur_model_id) {
    AsyncSendRegstMsgToProducer(model_rq.front());
    model_rq.pop();
  }
  if (!model_rq.empty()) {
    CHECK_EQ(model_rq.front()->model_version_id(), cur_model_id);
  }
}

void BackwardCompActor::AsyncReturnModelRegstUntilLastPieceIdGreaterThan(
    int64_t piece_id, std::queue<Regst*>& model_rq) {
  while (model_rq.empty() == false) {
    int64_t model_id = model_rq.front()->model_version_id();
    int64_t last_piece_id = GetLastPieceIdForModelVersionId(model_id);
    if (last_piece_id > piece_id) { return; }
    AsyncSendRegstMsgToProducer(model_rq.front());
    model_rq.pop();
  }
}

}  // namespace oneflow
