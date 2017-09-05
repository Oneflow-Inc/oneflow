#ifndef ONEFLOW_CORE_SCHEDULE_LONGEST_PATH_VISITOR_H_
#define ONEFLOW_CORE_SCHEDULE_LONGEST_PATH_VISITOR_H_

#include "oneflow/core/schedule/bfs_visitor.h"

namespace oneflow {
namespace schedule {

template<typename NodeType>
class LongestPathVisitor final {
 public:
  typedef std::function<void(NodeType)> NodeVisitor;
  typedef std::function<void(NodeType, const NodeVisitor&)> ForEachNode;

  typedef std::function<bool(NodeType, NodeType)> AncendenceIndicator;
  typedef std::function<float(NodeType)> NodeWeight;
  typedef std::function<void(const std::list<NodeType>&)> PathVisitor;
  typedef std::function<bool(const std::list<NodeType>&)> FinalPathIndicator;
  OF_DISALLOW_COPY_AND_MOVE(PathVisitor);
  PathVisitor(const ForEachNode& foreach_next,
				const ForEachNode& foreach_prev, const AncendenceIndicator& is_asc)
      : foreach_next_(foreach_next), foreach_prev_(foreach_prev),
				is_ancendent_(is_asc) {}
  ~PathVisitor() = default;

  uint32_t operator()(NodeType start,
											const NodeWeight& get_node_weight,
											const FinalPathIndicator& is_final_path,
											const PathVisitor& visitor) {
    return WalkPath(start, get_node_weight, is_final_path, visitor);
  }


 private:
  uint32_t WalkPath(NodeType start,
										const NodeWeight& get_node_weight,
										const FinalPathIndicator& is_final_path,
										const PathVisitor& path_visitor) {
		std::list<NodeType> starts{start};
		std::unordered_map<NodeType,
					std::unordered_map<NodeType, std::list<NodeType>>> start2end2path;
		std::unordered_map<NodeType,
					std::unordered_map<NodeType, float>> start2end2weight;
		auto bfs_foreach_prev = [&](NodeType node, const NodeVistor& cb) {
			foreach_prev_(node, [&](NodeType prev){
				if (prev == start || is_ancendent_(start, prev)) { cb(prev); }
			});
		};
		auto is_final_node = [&](NodeType node){
			const auto& l = start2end2path[start][node];
			return is_final_path(l);
		};
		auto node_visitor = [&](NodeType node){
			float max_path_weight = 0;
			NodeType max_path_prev = reinterpret_cast<NodeType>(0);
			if (node != start) {
				bfs_foreach_prev(node, [&](NodeType prev){
					float prev_path_weight = start2end2weight[start][prev];
					if (max_path_weight < prev_path_weight) {
						max_path_weight = prev_path_weight;
						max_path_prev = prev;
					}
				});
			}
			start2end2weight[start][node] = max_path_weight + get_node_weight(node);
			auto& path = start2end2path[start][node];
			path = start2end2path[start][max_path_prev];
			path.push_back(node);
			path_visitor(path);
		};
		BfsVistor<NodeType> bfs_visitor(foreach_next_, bfs_foreach_prev);
		bfs_visitor.Walk(start, is_final_node, node_visitor);
	}

  ForEachNode foreach_next_;
  ForEachNode foreach_prev_;
	AncendenceIndicator is_ancendent_;
};

}
}
#endif 	// ONEFLOW_CORE_SCHEDULE_LONGEST_PATH_VISITOR_H_
