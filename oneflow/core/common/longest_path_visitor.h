#ifndef ONEFLOW_CORE_COMMON_LONGEST_PATH_VISITOR_H_
#define ONEFLOW_CORE_COMMON_LONGEST_PATH_VISITOR_H_

#include "oneflow/core/common/bfs_visitor.h"

namespace oneflow {

template<typename NodeType>
class LongestPathVisitor final {
 public:
  typedef std::function<void(NodeType)> NodeVisitor;
  typedef std::function<void(NodeType, const NodeVisitor&)> ForEachNode;

  typedef std::function<bool(NodeType, NodeType)> AncendenceIndicator;
  typedef std::function<double(NodeType)> NodeWeight;
  typedef std::function<void(const std::list<NodeType>&)> PathVisitor;
  OF_DISALLOW_COPY_AND_MOVE(LongestPathVisitor);
  explicit LongestPathVisitor(const ForEachNode& foreach_next,
                              const ForEachNode& foreach_prev,
                              const AncendenceIndicator& is_asc)
      : foreach_next_(foreach_next),
        foreach_prev_(foreach_prev),
        is_ancendent_(is_asc) {}
  ~LongestPathVisitor() = default;

  uint32_t operator()(NodeType src_node, NodeType dst_node,
                      const NodeWeight& get_node_weight,
                      const PathVisitor& visitor) const {
    return WalkPath(src_node, dst_node, get_node_weight, visitor);
  }

 private:
  uint32_t WalkPath(NodeType src_node, NodeType dst_node,
                    const NodeWeight& get_node_weight,
                    const PathVisitor& path_visitor) const {
    std::list<NodeType> starts{src_node};
    std::unordered_map<NodeType, std::list<NodeType>> end2path;
    std::unordered_map<NodeType, double> end2weight;
    auto bfs_foreach_next = [&](NodeType node, const NodeVisitor& cb) {
      foreach_next_(node, [&](NodeType next) {
        if (next == dst_node || is_ancendent_(next, dst_node)) { cb(next); }
      });
    };
    auto bfs_foreach_prev = [&](NodeType node, const NodeVisitor& cb) {
      foreach_prev_(node, [&](NodeType prev) {
        if (prev == src_node || is_ancendent_(src_node, prev)) { cb(prev); }
      });
    };
    auto is_final_node = [&](NodeType node) { return node == dst_node; };
    auto node_visitor = [&](NodeType node) {
      double max_path_weight = 0;
      NodeType max_path_prev = reinterpret_cast<NodeType>(0);
      if (node != src_node) {
        bfs_foreach_prev(node, [&](NodeType prev) {
          double prev_path_weight = end2weight[prev];
          if (max_path_weight < prev_path_weight) {
            max_path_weight = prev_path_weight;
            max_path_prev = prev;
          }
        });
      }
      end2weight[node] = max_path_weight + get_node_weight(node);
      auto& path = end2path[node];
      path = end2path[max_path_prev];
      path.push_back(node);
      path_visitor(path);
    };
    BfsVisitor<NodeType> bfs_visitor(bfs_foreach_next, bfs_foreach_prev);
    return bfs_visitor.Walk(starts, is_final_node, node_visitor);
  }

  ForEachNode foreach_next_;
  ForEachNode foreach_prev_;
  AncendenceIndicator is_ancendent_;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_COMMON_LONGEST_PATH_VISITOR_H_
