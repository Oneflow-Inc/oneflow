#ifndef ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class BackwardCompActor : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BackwardCompActor);
  BackwardCompActor() = default;
  ~BackwardCompActor() = default;

 protected:
  int64_t is_out_diff_eord() const { return is_out_diff_eord_; }
  int64_t model_regst_desc_id() const { return model_regst_desc_id_; }
  int64_t model_tmp_regst_desc_id() const { return model_tmp_regst_desc_id_; }
  int64_t activation_regst_desc_id() const { return activation_regst_desc_id_; }
  int64_t data_tmp_regst_desc_id() const { return data_tmp_regst_desc_id_; }
  int64_t out_regst_desc_id() const { return out_regst_desc_id_; }
  int64_t in_regst_desc_id() const { return in_regst_desc_id_; }
  int64_t out_diff_regst_desc_id() const { return out_diff_regst_desc_id_; }
  ColIdOrder order() const { return order_; }
  Regst* model_tmp_regst() const { return model_tmp_regst_; }
  std::queue<Regst*>* model_regsts() { return &model_regsts_; }
  bool has_cur_piece_started() const { return has_cur_piece_started_; }

  void set_has_cur_piece_started(bool val) { has_cur_piece_started_ = val; }
  void set_is_out_diff_eord(bool val) { is_out_diff_eord_ = val; }
  void set_model_tmp_regst(Regst* regst) { model_tmp_regst_ = regst; }

  HashMap<int64_t, std::deque<std::deque<Regst*>>>* readable_deq_regsts() {
    return &readable_deq_regsts_;
  }

  void HandleOutDiffRegsts(Regst*, std::deque<std::deque<Regst*>>*);
  void AsyncReturnModelRegstUntilMatchCurOutRegst(int64_t cur_model_id);
  void AsyncReturnModelRegstUntilLastPieceIdGreaterThan(int64_t piece_id);
  void ForCurReadableModelAndModelTmp(
      std::function<void(const Regst*)> handler);

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void AsyncReturnAllReadableRegst() override;
  virtual void VirtualBackwardCompActorInit(const TaskProto&) = 0;
  virtual void CheckBeforeAsyncReturnAllReadableRegst() = 0;
  void TryAsyncReturnModelRegst();
  void TryAsyncReturnModelTmpRegst();

  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  int64_t activation_regst_desc_id_;
  int64_t data_tmp_regst_desc_id_;
  int64_t out_regst_desc_id_;
  int64_t in_regst_desc_id_;
  int64_t out_diff_regst_desc_id_;

  bool is_out_diff_eord_;
  ColIdOrder order_;

  std::queue<Regst*> model_regsts_;
  Regst* model_tmp_regst_;
  bool has_cur_piece_started_;

  HashMap<int64_t, std::deque<std::deque<Regst*>>> readable_deq_regsts_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_
