#ifndef ONEFLOW_CORE_ACTOR_FORWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_FORWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class ForwardCompActor : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForwardCompActor);
  ForwardCompActor() = default;
  ~ForwardCompActor() = default;

 protected:
  bool is_in_eord() const { return is_in_eord_; }
  int64_t in_regst_desc_id() const { return in_regst_desc_id_; }
  int64_t model_regst_desc_id() const { return model_regst_desc_id_; }
  int64_t model_tmp_regst_desc_id() const { return model_tmp_regst_desc_id_; }
  Regst* model_tmp_regst() const { return model_tmp_regst_; }

  void set_is_in_eord(bool val) { is_in_eord_ = val; }
  void set_model_tmp_regst(Regst* val) { model_tmp_regst_ = val; }

  void SwitchToHandlerInitModelTmpOrNormal();
  int HandlerInitModel(const ActorMsg&);
  int HandlerInitModelTmp(const ActorMsg&);
  void TryAsyncReturnModelTmpRegst();

 private:
  void VirtualCompActorInit(const TaskProto& task_proto) override;
  void AsyncReturnAllReadableRegst() override;
  virtual void VirtualForwardCompActorInit(const TaskProto&) = 0;
  virtual void TryAsyncReturnModelRegst() = 0;
  virtual void CheckBeforeAsyncReturnAllReadableRegst() = 0;

  bool is_in_eord_;
  int64_t in_regst_desc_id_;
  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  Regst* model_tmp_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_FORWARD_COMPUTE_ACTOR_H_
