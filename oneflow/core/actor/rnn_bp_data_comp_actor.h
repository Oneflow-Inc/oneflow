#ifndef ONEFLOW_CORE_ACTOR_BP_DATA_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_BP_DATA_COMP_ACTOR_H_

#include <list>
#include <stack>
#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class RnnBpDataCompActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RnnBpDataCompActor);
  RnnBpDataCompActor() = default;
  ~RnnBpDataCompActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;

 private:
  int HandlerNormal(const ActorMsg&) override;
  int HandlerWaitUntilNoReadableRegst(const ActorMsg&) override;

  bool IsReadReady() override;
  void Act() override;
  void AsyncSendMsgToModelProducer();

  bool CheckModel_Out_OutDiff(Regst* cur_regst) const;
  void FillMatl4ActWithIn_Out_OutDiff_Model(Regst* cur_regst);

  CudaStreamHandle cuda_handle_;

  int64_t in_regst_desc_id_;
  HashMap<int64_t, std::stack<Regst*>> pid2in_regsts_;

  int64_t out_regst_desc_id_;
  HashMap<int64_t, std::list<Regst*>> pid2out_regsts_;

  int64_t initial_hidden_regst_desc_id_;
  HashMap<int64_t, Regst*> pid2init_hid_regsts_;
  int64_t expected_initial_hidden_piece_id_;

  int64_t out_diff_regst_desc_id_;
  // regst in deque is ascending by col_id
  HashMap<int64_t, std::deque<Regst*>> pid2out_diff_regsts_;
  bool is_insert_to_back_;

  int64_t rec_acc_diff_regst_desc_id_;  // recurrent_accumulate_diff
  HashMap<int64_t, Regst*> pid2rec_acc_diff_regsts_;

  int64_t model_regst_desc_id_;
  HashMap<int64_t, Regst*> model_vid2model_regst_;
  HashMap<int64_t, int64_t> model_vid2cnt_;
  // <model_version_id, no_more_new_piece>, default as false
  HashMap<int64_t, bool> model_vid2status_;
  int64_t expected_model_version_id_;

  struct Material4Act {
    Material4Act() : readable_regsts_(), pre_out_regst(nullptr) {}

    HashMap<int64_t, Regst*> readable_regsts_;
    Regst* pre_out_regst;  // pre_out_regst & out_regst have same regst_desc_id
  };
  Material4Act matl4act_;
};

}  // namespace oneflow

#endif
