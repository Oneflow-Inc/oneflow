#ifndef ONEFLOW_CORE_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class MdUpdtCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtCompActor);
  MdUpdtCompActor() = default;
  ~MdUpdtCompActor() = default;

  void Init(const TaskProto&) override;
  void ProcessMsg(const ActorMsg&, const ThreadContext&) override;

 private:
  class State;
  class BeforeInitializeModelState;
  class BeforeSendInitialModelState;
  class UpdateModelState;
  class EndState;

  State* state_;
  uint64_t model_regst_desc_id_;
  uint64_t model_tmp_regst_desc_id_;

};

class MdUpdtCompActor::State {
 public:
  OF_DISALLOW_COPY_AND_MOVE(State);
  virtual ~State() = default;

  virtual void ProcessMsg(const ActorMsg&,
                          const KernelContext&,
                          MdUpdtCompActor* actor) = 0;

 protected:
  State() = default;

};

class MdUpdtCompActor::BeforeInitializeModelState final : public State {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BeforeInitializeModelState);
  ~BeforeInitializeModelState() = default;

  OF_SINGLETON(BeforeInitializeModelState);

  void ProcessMsg(const ActorMsg&,
                  const KernelContext&,
                  MdUpdtCompActor* actor) override;

 private:
  BeforeInitializeModelState() = default;

};

class MdUpdtCompActor::BeforeSendInitialModelState final : public State {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BeforeSendInitialModelState);
  ~BeforeSendInitialModelState() = default;

  OF_SINGLETON(BeforeSendInitialModelState);

  void ProcessMsg(const ActorMsg&,
                  const KernelContext&,
                  MdUpdtCompActor* actor) override;

 private:
  BeforeSendInitialModelState() = default;
};

class MdUpdtCompActor::UpdateModelState final : public State {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UpdateModelState);
  ~UpdateModelState() = default;

  OF_SINGLETON(UpdateModelState);

  void ProcessMsg(const ActorMsg&,
                  const KernelContext&,
                  MdUpdtCompActor* actor) override;

 private:
  UpdateModelState() = default;
};

class MdUpdtCompActor::EndState final : public State {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EndState);
  ~EndState() = default;

  OF_SINGLETON(EndState);

  void ProcessMsg(const ActorMsg&,
                  const KernelContext&,
                  MdUpdtCompActor* actor) override;

 private:
  EndState() = default;

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_
