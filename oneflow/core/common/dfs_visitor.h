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
    Walk(start_nodes, Handler, [](NodeType) {});
  }

  void operator()(const std::list<NodeType>& start_nodes,
                  const NodeHandlerFn& OnEnter, const NodeHandlerFn& OnExit) {
    Walk(start_nodes, OnEnter, OnExit);
  }

 private:
  void Walk(const std::list<NodeType>& start_nodes,
            const NodeHandlerFn& OnEnter, const NodeHandlerFn& OnExit) {
    HashMap<NodeType, bool> visited;
    std::stack<std::list<NodeType>> stack;
    stack.push(start_nodes);
    while (true) {
      if (!stack.top().empty()) {
        NodeType node = stack.top().front();
        if (!visited[node]) {
          OnEnter(node);
          visited[node] = true;
          stack.push({});
          foreach_next_(node,
                        [&](NodeType next) { stack.top().push_back(next); });
        } else {
          stack.top().erase(stack.top().begin());
        }
      } else {
        stack.pop();
        if (stack.empty()) { break; }
        OnExit(stack.top().front());
        stack.top().erase(stack.top().begin());
      }
    }
  }

  ForEachNodeFn foreach_next_;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_COMMON_DFS_VISITOR_H_
