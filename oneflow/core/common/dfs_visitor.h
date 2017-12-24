#ifndef ONEFLOW_CORE_COMMON_DFS_VISITOR_H_
#define ONEFLOW_CORE_COMMON_DFS_VISITOR_H_

#include <stack>
#include "oneflow/core/common/util.h"

namespace oneflow {

//  depth first search visitor
template<typename NodeType>
class DfsVisitor final {
 public:
  typedef std::function<void(NodeType)> NodeHandlerFn;
  typedef std::function<void(NodeType, const NodeHandlerFn&)> ForEachNodeFn;

  OF_DISALLOW_COPY_AND_MOVE(DfsVisitor);
  DfsVisitor(const ForEachNodeFn& ForEachNext) : foreach_next_(ForEachNext) {}
  virtual ~DfsVisitor() = default;

  void operator()(const std::list<NodeType>& start_nodes,
                  const NodeHandlerFn& Handler) {
    HashMap<NodeType, bool> visited_or_visiting_soon;
    std::stack<NodeType> stack;
    for (NodeType node : start_nodes) {
      stack.push(node);
      visited_or_visiting_soon[node] = true;
    }
    while (!stack.empty()) {
      NodeType node = stack.top();
      Handler(node);
      stack.pop();
      foreach_next_(node, [&](NodeType next) {
        if (!visited_or_visiting_soon[next]) {
          stack.push(next);
          visited_or_visiting_soon[next] = true;
        }
      });
    }
  }

 private:
  ForEachNodeFn foreach_next_;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_COMMON_DFS_VISITOR_H_
