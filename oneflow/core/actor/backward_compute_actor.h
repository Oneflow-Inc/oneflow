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

  void set_is_out_diff_eord(bool val) { is_out_diff_eord_ = val; }

  void AsyncReturnModelRegstUntilMatchCurOutRegst(int64_t cur_model_id,
                                                  std::queue<Regst*>& model_rq);
  void AsyncReturnModelRegstUntilLastPieceIdGreaterThan(
      int64_t piece_id, std::queue<Regst*>& model_rq);

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  virtual void VirtualBackwardCompActorInit(const TaskProto&) = 0;

  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  int64_t activation_regst_desc_id_;
  int64_t data_tmp_regst_desc_id_;
  int64_t out_regst_desc_id_;
  int64_t in_regst_desc_id_;
  int64_t out_diff_regst_desc_id_;
  bool is_out_diff_eord_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_
