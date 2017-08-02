#ifndef ONEFLOW_CORE_SCHEDULE_IMPLEMENT_NAIVE_H_
#define ONEFLOW_CORE_SCHEDULE_IMPLEMENT_NAIVE_H_

#include "oneflow/core/schedule/interface/policy.h"
#include "oneflow/core/schedule/util/util.h"

namespace oneflow {
namespace schedule {

class TestGraphGeneratorNaivePolicy : public TestGraphGeneratorPolicy {
 public:
  POLICY_IMPLEMENT_BOILERPLATE(TestGraphGeneratorNaivePolicy,
                               TestGraphGeneratorPolicy);

  virtual std::unique_ptr<GraphNode> Demo();
};

class PrinterNaivePolicy : public PrinterPolicy {
 public:
  POLICY_IMPLEMENT_BOILERPLATE(PrinterNaivePolicy, PrinterPolicy);
  virtual void PrintGraph(const GraphNode& graph, const std::string& filename);
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_IMPLEMENT_NAIVE_H_
