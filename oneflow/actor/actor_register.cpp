#include "actor/actor_register.h"
#include "glog/logging.h"

namespace oneflow {

namespace {

struct pair_hash {
  std::size_t operator () (const std::pair<int, bool> &p) const {
    return std::hash<int>{}((p.first << 1) | ((int)p.second));
  }
};

using ActorTypePair = std::pair<TaskType, bool>;
using ActorCreatorMap = HashMap<ActorTypePair, std::function<Actor*()>, pair_hash>;

ActorCreatorMap& ActorType2Creator() {
  static ActorCreatorMap obj;
  return obj;
}

}

void AddActorCreator(TaskType task_type, bool is_forward,
                     std::function<Actor*()> creator) {
  CHECK(ActorType2Creator().emplace(std::make_pair(task_type, is_forward), creator).second);
}

std::shared_ptr<Actor> ConstructActor(const TaskProto& task_proto) {
  ActorTypePair actor_type_pair = std::make_pair(task_proto.type(),
                                                 task_proto.is_forward());
  std::shared_ptr<Actor> ret(ActorType2Creator().at(actor_type_pair)());
  ret->Init(task_proto);
  return ret;
}

}  // namespace oneflow
