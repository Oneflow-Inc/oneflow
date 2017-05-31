#include "actor/actor_registry.h"
#include "glog/logging.h"

namespace oneflow {

namespace {

struct PairHash {
  std::size_t operator () (const std::pair<int, bool> &p) const {
    return std::hash<int>{}((p.first << 1) | (static_cast<int>(p.second)));
  }
};

using ActorTypePair = std::pair<TaskType, bool>;
using ActorCreatorMap = HashMap<ActorTypePair, std::function<Actor*()>, PairHash>;

ActorCreatorMap& ActorType2Creator() {
  static ActorCreatorMap obj;
  return obj;
}

}

void AddActorCreator(TaskType task_type, bool is_forward,
                     std::function<Actor*()> creator) {
  ActorTypePair actor_type_pair{task_type, is_forward};
  CHECK(ActorType2Creator().emplace(actor_type_pair, creator).second);
}

std::shared_ptr<Actor> ConstructActor(const TaskProto& task_proto) {
  ActorTypePair actor_type_pair{task_proto.type(), task_proto.is_forward()};
  std::shared_ptr<Actor> ret(ActorType2Creator().at(actor_type_pair)());
  ret->Init(task_proto);
  return ret;
}

}  // namespace oneflow
