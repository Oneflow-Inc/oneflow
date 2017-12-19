#include "oneflow/core/job/improver.h"
#include "oneflow/core/graph/ss_task_graph.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

namespace {

void ParseActEvents(const std::string& act_event_filepath,
		    std::list<ActEvent>* act_events) {
  NormalPersistentInStream in_stream(LocalFS(), act_event_filepath);
  size_t act_event_size;
  while (!in_stream.Read(static_cast<char*>(&act_event_size, sizeof(size_t)))) {
    std::vector<char> buffer(act_event_size);
    CHECK(!in_stream.Read(buffer.data(), act_event_size));
    ActEvent act_event;
    act_event.ParseFromArray(buffer.data(), act_event_size);
    act_events->push_back();
  }
}

}

Plan Improver::Improve(const Plan& naive_plan,
                       const std::string& act_event_filepath) {
  std::list<ActEvent> act_events;
  ParseActEvents(act_event_filepath, &act_events);
  SSTaskGraph ss_task_graph(naive_plan, act_events);
}

}  // namespace oneflow
