#ifndef ONEFLOW_CORE_ACTOR_NAIVE_READABLE_REGISTER_MANAGER_H_
#define ONEFLOW_CORE_ACTOR_NAIVE_READABLE_REGISTER_MANAGER_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class NaiveReadableRegstMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NaiveReadableRegstMgr);
  NaiveReadableRegstMgr() : readable_regst_cnt_(0) {}
  ~NaiveReadableRegstMgr() = default;

  void Init(const TaskProto& task_proto);
  void Push(Regst* regst);
  void ReturnToProducerAndPopCurReadable(Actor* actor, std::function<bool(Regst*)> IsAllowed);
  void ReturnToProducerAndPopCurReadable(Actor* actor);
  Regst* GetCurReadable(int64_t regst_desc_id);
  Regst* GetFirstCurReadable() { return readable_regst_.begin()->second.front(); }
  void ForEachCurReadableRegst(std::function<void(Regst*)> func);
  bool IsReadReady();
  bool IsEmpty() { return readable_regst_cnt_ == 0; }

 private:
  HashMap<int64_t, std::queue<Regst*>> readable_regst_;  // regst_desc_id
  size_t readable_regst_cnt_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_NAIVE_READABLE_REGISTER_MANAGER_H_
