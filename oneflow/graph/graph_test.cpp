#include "oneflow/graph/graph.h"
#include <vector>
#include <map>
#include <unordered_set>
#include <utility>
#include <memory>
#include "gtest/gtest.h"
#include "oneflow/common/util.h"
#include "oneflow/graph/node.h"

namespace oneflow {

class TestEdge;

class TestNode final : public Node<TestNode, TestEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TestNode);
  TestNode(uint64_t node_id_) {
    test_node_id_ = node_id_;
  }
  ~TestNode() = default;

  uint64_t test_node_id() const { return test_node_id_; }
 private:
  uint64_t test_node_id_;
};

class TestEdge final : public Edge<TestNode, TestEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TestEdge);
  TestEdge() = default;
  ~TestEdge() = default;

 private:
};

class TestGraph final : public Graph<TestNode, TestEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TestGraph);
  TestGraph() = delete;
  ~TestGraph() = default;

  TestGraph(const std::vector<std::vector<uint64_t>>& graph_conf) {
    std::vector<TestNode*> node_id2node;
    for (size_t i = 0; i < graph_conf.size(); ++i) {
      TestNode* cur_node = new TestNode(i);
      EnrollNode(cur_node);
      node_id2node.push_back(cur_node);
    }
    for (size_t i = 0; i < graph_conf.size(); ++i) {
      TestNode* src_node = node_id2node[i];
      for (size_t j = 0; j < graph_conf[i].size(); ++j) {
        TestEdge* edge = NewEdge();
        TestNode* dst_node = node_id2node[graph_conf[i][j]];
        Connect(src_node, edge, dst_node);
      }
    }
    UpdateSourceAndSink();
  }
};

using NodeIdPair = std::pair<uint64_t, uint64_t>;

void DoOneTestGraph(const TestGraph& test_graph,
                    const std::vector<std::vector<uint64_t>>& graph_conf) {
  uint64_t node_num = graph_conf.size();
  // 1. Determines whether the traversal result satisfies the topological order
  std::vector<uint64_t> topo_array;
  HashMap<uint64_t, uint64_t> node_id2order;
  auto NodePairHash = [](const NodeIdPair& val) { return val.first ^ val.second; };
  std::unordered_set<NodeIdPair,
                     decltype(NodePairHash)> edges_node_pair(10, NodePairHash);
  uint64_t order = 0;
  for (auto it = test_graph.cbegin(); it != test_graph.cend(); ++it) {
    topo_array.push_back(it->test_node_id());
    node_id2order.emplace(it->test_node_id(), order);
    ++order;
  }
  ASSERT_EQ(topo_array.size(), node_num);
  // method : 
  // judge every edge <u,v>
  // the node u's order in topo_array is smaller than node v
  uint64_t edge_num = 0;
  for (uint64_t src_node_id = 0; src_node_id < node_num; ++src_node_id) {
    uint64_t src_node_order = node_id2order.at(src_node_id);
    for (uint64_t dst_node_id : graph_conf[src_node_id]) {
      uint64_t dst_node_order = node_id2order.at(dst_node_id);
      ASSERT_LT(src_node_order, dst_node_order);
      ++edge_num;
      edges_node_pair.insert(std::make_pair(src_node_id, dst_node_id));
    }
  }
  // 2. judge whether the getter method of Graph can return all nodes and edges
  ASSERT_EQ(test_graph.nodes().size(), node_num);
  ASSERT_EQ(test_graph.edges().size(), edge_num);
  std::unordered_set<uint64_t> node_ids;
  for (const std::unique_ptr<TestNode>& cur_node : test_graph.nodes()) {
    uint64_t cur_node_id = cur_node->test_node_id();
    ASSERT_TRUE(node_ids.insert(cur_node_id).second);
    ASSERT_LT(cur_node_id, node_num);
    ASSERT_GE(cur_node_id, 0);
  }
  for (const std::unique_ptr<TestEdge>&  cur_edge : test_graph.edges()) {
    uint64_t src_node_id = cur_edge->src_node()->test_node_id();
    uint64_t dst_node_id = cur_edge->dst_node()->test_node_id();
    ASSERT_TRUE(
        edges_node_pair.count(std::make_pair(src_node_id, dst_node_id)) > 0);
  }
}

TEST(TestGraph, test_graph_node_num_7) {
  std::vector<std::vector<uint64_t>> graph_conf;
  for (uint64_t i = 0; i < 7; ++i) {
    graph_conf.push_back(std::vector<uint64_t>());
  }
  graph_conf[2].push_back(1);
  graph_conf[2].push_back(0);
  graph_conf[2].push_back(3);
  graph_conf[1].push_back(0);
  graph_conf[0].push_back(4);
  graph_conf[5].push_back(6);
  TestGraph test_graph(graph_conf);
  DoOneTestGraph(test_graph, graph_conf);
}

}// namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
