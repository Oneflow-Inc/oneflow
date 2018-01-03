#ifndef ONEFLOW_CORE_COMMON_DAG_TOPO_VISITOR_H_
#define ONEFLOW_CORE_COMMON_DAG_TOPO_VISITOR_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename NodeType>
class DAGTopoVisitor final {
 public:
  typedef std::function<void(NodeType)> NodeHandlerFn;
  typedef std::function<bool(NodeType)> FinalNodeIndicatorFn;
  typedef std::function<void(NodeType, const NodeHandlerFn&)> ForEachNodeFn;

  OF_DISALLOW_COPY_AND_MOVE(DAGTopoVisitor);
  DAGTopoVisitor(const ForEachNodeFn& foreach_next,
                 const ForEachNodeFn& foreach_prev)
      : foreach_next_(foreach_next), foreach_prev_(foreach_prev) {}
  virtual ~DAGTopoVisitor() = default;

  void operator()(const std::list<NodeType>& starts,
                  const NodeHandlerFn& visitor) const {
    Walk(starts, [](NodeType) { return false; }, visitor);
  }

  void operator()(const std::list<NodeType>& starts,
                  const FinalNodeIndicatorFn& is_final,
                  const NodeHandlerFn& visitor) const {
    Walk(starts, is_final, visitor);
  }

 private:
  void Walk(const std::list<NodeType>& init_nodes,
            const FinalNodeIndicatorFn& is_final,
	    const NodeHandlerFn& cb) const {
    CheckInitNodesHasNoPrev(init_nodes);

    HashMap<NodeType, bool> has_queued;
    std::queue<NodeType> queue;
    for (NodeType node : init_nodes) {
      queue.push(node);
      has_queued[node] = true;
    }
    while (queue.size()) {
      NodeType node = queue.front();
      cb(node);
      if (is_final(node)) break;
      queue.pop();
      foreach_next_(node, [&](NodeType next) {
        bool all_prev_has_queued = true;
        foreach_prev_(next, [&](NodeType prev) {
          if (all_prev_has_queued && !has_queued[prev]) {
            all_prev_has_queued = false;
          }
        });
        if (all_prev_has_queued && !has_queued[next]) {
          queue.push(next);
          has_queued[next] = true;
        }
      });
    }
  }

  void CheckInitNodesHasNoPrev(const std::list<NodeType>& init_nodes) const {
    for (NodeType node : init_nodes) {
      int prev_cnt = 0;
      foreach_prev_(node, [&](NodeType) { ++prev_cnt; });
      CHECK(prev_cnt == 0);
    }
  }

  ForEachNodeFn foreach_next_;
  ForEachNodeFn foreach_prev_;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_COMMON_DAG_TOPO_VISITOR_H_
