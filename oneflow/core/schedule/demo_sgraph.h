#ifndef ONEFLOW_CORE_SCHEDULE_DEMO_SGRAPH_H_
#define ONEFLOW_CORE_SCHEDULE_DEMO_SGRAPH_H_

#include "oneflow/core/schedule/sgraph.h"

namespace oneflow {
namespace schedule {

class DemoSGraph final : public SGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DemoSGraph);
  explicit DemoSGraph(const std::string& name) : SGraph(name) {
    InitSourceAndSink();
    InitFromDemo();
    Update();
  }
  explicit DemoSGraph(const Plan& plan) : SGraph(plan) {
    InitSourceAndSink();
    InitFromDemo();
    Update();
  }

 private:
  void InitFromDemo();
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_DEMO_SGRAPH_H_
