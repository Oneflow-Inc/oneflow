#ifndef ONEFLOW_CORE_SCHEDULE_BFS_VISITOR_H_
#define ONEFLOW_CORE_SCHEDULE_BFS_VISITOR_H_

#include <queue>

#include "oneflow/core/schedule/snode.h"

namespace oneflow {
namespace schedule {

template<typename NodeType>
class BfsVisitor final {
 public:
  typedef std::function<void(NodeType)> NodeVisitor;
  typedef std::function<void(NodeType, const NodeVisitor&)> ForEachNode;

  OF_DISALLOW_COPY_AND_MOVE(BfsVisitor);
  BfsVisitor(const ForEachNode& foreach_prev, const ForEachNode& foreach_next)
      : foreach_prev_(foreach_prev), foreach_next_(foreach_next) {}
  ~BfsVisitor() = default;

  uint32_t operator()(NodeType start, const NodeVisitor& visitor) {
    return Walk(std::list<NodeType>{start}, visitor);
  }

  uint32_t operator()(const std::list<NodeType>& starts,
                      const NodeVisitor& visitor) {
    return Walk(starts, visitor);
  }

 private:
  uint32_t Walk(const std::list<NodeType>& init_nodes, const NodeVisitor& cb) {
    Reset();
    MarkAllPrevVisited(init_nodes);
    uint32_t cnt = 0;
    std::queue<NodeType> queue;
    for (auto node : init_nodes) { queue.push(node); }
    while (queue.size()) {
      NodeType node = queue.front();
      cb(node);
      ++cnt;
      queue.pop();
      foreach_next_(node, [&](NodeType next) {
        bool all_marked = true;
        foreach_prev_(next, [&](NodeType prev) {
          if (all_marked && !visited_or_visiting_soon_[prev]) {
            all_marked = false;
          }
        });
        if (all_marked && !visited_or_visiting_soon_[next]) {
          queue.push(next);
          visited_or_visiting_soon_[next] = true;
        }
      });
    }
    return cnt;
  }

  void Reset() { visited_or_visiting_soon_.clear(); }

  void MarkAllPrevVisited(const std::list<NodeType>& init_nodes) {
    std::queue<NodeType> queue;
    for (auto node : init_nodes) { queue.push(node); }
    while (queue.size()) {
      NodeType node = queue.front();
      visited_or_visiting_soon_[node] = true;
      queue.pop();
      foreach_prev_(node, [&](NodeType prev) {
        if (!visited_or_visiting_soon_[prev]) {
          queue.push(prev);
          visited_or_visiting_soon_[prev] = true;
        }
      });
    }
  }

  ForEachNode foreach_prev_;
  ForEachNode foreach_next_;
  std::unordered_map<NodeType, bool> visited_or_visiting_soon_;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_BFS_VISITOR_H_
