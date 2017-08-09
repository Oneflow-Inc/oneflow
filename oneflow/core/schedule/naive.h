#ifndef ONEFLOW_CORE_SCHEDULE_IMPLEMENT_NAIVE_H_
#define ONEFLOW_CORE_SCHEDULE_IMPLEMENT_NAIVE_H_

#include "oneflow/core/schedule/policy.h"
#include "oneflow/core/schedule/util.h"

namespace oneflow {
namespace schedule {

class TestGraphGeneratorNaivePolicy : public TestGraphGeneratorPolicy {
 public:
  POLICY_IMPLEMENT_BOILERPLATE(TestGraphGeneratorNaivePolicy,
                               TestGraphGeneratorPolicy);

  virtual std::unique_ptr<SGraph> DemoGraph();
};

class PrinterNaivePolicy : public PrinterPolicy {
 public:
  POLICY_IMPLEMENT_BOILERPLATE(PrinterNaivePolicy, PrinterPolicy);
  virtual void PrintGraph(const SGraph& graph, const std::string& filename);
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_IMPLEMENT_NAIVE_H_
