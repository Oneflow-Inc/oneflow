#ifndef ONEFLOW_CORE_GRAPH_GRAPH_VISITOR_UTIL_H_
#define ONEFLOW_CORE_GRAPH_GRAPH_VISITOR_UTIL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename NodeType>
class GraphNodeVisitorUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GraphNodeVisitorUtil);
  GraphNodeVisitorUtil() = delete;
  ~GraphNodeVisitorUtil() = delete;

  using HandlerType = std::function<void(NodeType)>;
  using ForEachFnType = std::function<void(NodeType, const HandlerType&)>;
  static void BfsForEach(const std::list<NodeType>& starts,
                         const ForEachFnType& ForEachNext,
                         const HandlerType& Handler) {
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

  static void TopoForEach(const std::list<NodeType>& starts,
                          const ForEachFnType& ForEachInNode,
                          const ForEachFnType& ForEachOutNode,
                          const HandlerType& Handler) {
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

#endif  // ONEFLOW_CORE_GRAPH_GRAPH_VISITOR_UTIL_H_
