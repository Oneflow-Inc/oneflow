#ifndef ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_

#include "oneflow/core/actor/copy_actor.h"

namespace oneflow {

class CopyHdActor final : public CopyActor {
public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdActor);
  CopyHdActor() = default;
  ~CopyHdActor() = default;

private:

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_
