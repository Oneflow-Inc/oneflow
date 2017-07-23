#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

namespace {

struct PairHash {
  std::size_t operator()(const std::pair<int, bool>& p) const {
    return std::hash<int>{}((p.first << 1) | (static_cast<int>(p.second)));
  }
};

using ActorTypePair = std::pair<TaskType, bool>;
using ActorCreatorMap =
    HashMap<ActorTypePair, std::function<Actor*()>, PairHash>;

ActorCreatorMap& ActorType2Creator() {
  static ActorCreatorMap obj;
  return obj;
}

}  // namespace

void AddActorCreator(TaskType task_type, bool is_forward,
                     std::function<Actor*()> creator) {
  ActorTypePair actor_type_pair{task_type, is_forward};
  CHECK(ActorType2Creator().emplace(actor_type_pair, creator).second);
}

std::unique_ptr<Actor> ConstructActor(const TaskProto& task_proto,
                                      const ThreadCtx& thread_ctx) {
  ActorTypePair actor_type_pair{task_proto.type(), task_proto.is_forward()};
  auto it = ActorType2Creator().find(actor_type_pair);
  CHECK(it != ActorType2Creator().end()) << task_proto.DebugString();
  std::unique_ptr<Actor> ret(it->second());
  ret->Init(task_proto, thread_ctx);
  return ret;
}

}  // namespace oneflow
