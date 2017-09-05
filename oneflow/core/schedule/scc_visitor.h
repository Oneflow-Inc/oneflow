#ifndef ONEFLOW_CORE_SCHEDULE_SCC_VISITOR_H_
#define ONEFLOW_CORE_SCHEDULE_SCC_VISITOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/schedule/snode.h"

namespace oneflow {
namespace schedule {

//	strongly connected component visitor
template<typename NodeType>
class SccVisitor final {
 public:
  typedef std::function<void(NodeType)> NodeVisitor;
  typedef std::function<void(const std::list<NodeType>&)> ComponentVisitor;
  typedef std::function<void(NodeType, const NodeVisitor&)> ForEachNode;

  OF_DISALLOW_COPY_AND_MOVE(SccVisitor);
  SccVisitor(const ForEachNode& foreach_next) : foreach_next_(foreach_next) {}
  ~SccVisitor() = default;

  uint32_t operator()(NodeType start,
                      const ComponentVisitor& component_visitor) {
    Reset();
    Walk(start, component_visitor);
    return component_cnt_;
  }
  uint32_t operator()(NodeType start) {
    return (*this)(start, [](const std::list<NodeType>&) {});
  }

  uint32_t operator()(const std::list<NodeType>& starts,
                      const ComponentVisitor& component_visitor) {
    Reset();
    for (NodeType start : starts) { Walk(start, component_visitor); }
    return component_cnt_;
  }

  uint32_t operator()(const std::list<NodeType>& starts) {
    return (*this)(starts, [](const std::list<NodeType>&) {});
  }

 private:
  void Reset() {
    node2index_.clear();
    node2low_.clear();
    stack_.clear();
    index_ = 0u;
    component_cnt_ = 0u;
  }

  void Walk(NodeType node, const ComponentVisitor& do_each_component) {
    if (node2index_[node]) { return; }
    node2index_[node] = node2low_[node] = ++index_;
    stack_.push_front(node);
    node2is_on_stack_[node] = true;
    bool is_self_loop = false;
    foreach_next_(node, [&](NodeType next) {
      if (!node2index_[next]) {
        Walk(next, do_each_component);
        node2low_[node] = std::min(node2low_[node], node2low_[next]);
      } else if (node2is_on_stack_[next]) {
        node2low_[node] = std::min(node2low_[node], node2low_[next]);
      }
      is_self_loop = (is_self_loop || node == next);
    });
    if (node2index_[node] == node2low_[node]) {
      std::list<NodeType> scc;
      NodeType w;
      do {
        w = stack_.front();
        stack_.pop_front();
        node2is_on_stack_[w] = false;
        scc.push_front(w);
      } while (w != node);
      if (scc.size() > 1 || is_self_loop) {
        do_each_component(scc);
        ++component_cnt_;
      }
    }
  }

  ForEachNode foreach_next_;
  std::unordered_map<NodeType, uint32_t> node2index_;
  std::unordered_map<NodeType, uint32_t> node2low_;
  std::unordered_map<NodeType, bool> node2is_on_stack_;
  std::list<NodeType> stack_;
  uint32_t index_ = 0u;
  uint32_t component_cnt_ = 0u;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SCC_VISITOR_H_
