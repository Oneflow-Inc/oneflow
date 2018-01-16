#ifndef ONEFLOW_CORE_GRAPH_VISITOR_H_
#define ONEFLOW_CORE_GRAPH_VISITOR_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename NodeType>
class Visitor final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Visitor);
  using NodeList = std::list<NodeType>;
  using NodeHandlerFn = std::function<void(NodeType)>;
  using ForEachNodeFn = std::function<void(NodeType, const NodeHandlerFn&)>;
  static void BfsForEach(const NodeList& starts,
                         const ForEachNodeFn& ForEachNext,
                         const NodeHandlerFn& Handler) {
    HashMap<NodeType, bool> has_queued;
    std::queue<NodeType> queue;
    for (NodeType start : starts) {
      queue.push(start);
      has_queued[start] = true;
    }
    while (!queue.empty()) {
      NodeType cur_node = queue.front();
      queue.pop();
      Handler(cur_node);
      ForEachNext(cur_node, [&](NodeType next) {
        if (!has_queued[next]) {
          queue.push(next);
          has_queued[next] = true;
        }
      });
    }
  }

  static void TopoForEach(const NodeList& starts,
                          const ForEachNodeFn& ForEachInNode,
                          const ForEachNodeFn& ForEachOutNode,
                          const NodeHandlerFn& Handler) {
    HashMap<NodeType, bool> has_queued;
    std::queue<NodeType> queue;
    for (NodeType start : starts) {
      queue.push(start);
      has_queued[start] = true;
      ForEachInNode(start, [&](NodeType) { CHECK(1) << "not a source"; });
    }
    while (!queue.empty()) {
      NodeType cur_node = queue.front();
      queue.pop();
      Handler(cur_node);
      ForEachOutNode(cur_node, [&](NodeType out) {
        bool will_be_ready = true;
        ForEachInNode(out, [&](NodeType in) {
          if (will_be_ready && !has_queued[in]) { will_be_ready = false; }
        });
        if (will_be_ready && !has_queued[out]) {
          queue.push(out);
          has_queued[out] = true;
        }
      });
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_VISITOR_H_
