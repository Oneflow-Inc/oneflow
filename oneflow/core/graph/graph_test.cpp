#include "oneflow/core/graph/graph.h"

namespace oneflow {

class TestEdge;

class TestNode final : public Node<TestNode, TestEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TestNode);
  TestNode(int64_t node_id_) {
    test_node_id_ = node_id_;
  }
  ~TestNode() = default;

  int64_t test_node_id() const { return test_node_id_; }
 private:
  int64_t test_node_id_;
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

  TestGraph(const std::vector<std::vector<int64_t>>& graph_conf) {
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

using NodeIdPair = std::pair<int64_t, int64_t>;

void DoOneTestGraph(const TestGraph& test_graph,
                    const std::vector<std::vector<int64_t>>& graph_conf) {
  int64_t node_num = graph_conf.size();

  // 1. Determines whether the traversal result satisfies the topological order
  HashMap<int64_t, int64_t> node_id2order, node_id2rorder;
  auto NodePairHash = [](const NodeIdPair& val) { return val.first ^ val.second; };
  std::unordered_set<NodeIdPair,
                     decltype(NodePairHash)> edges_node_pair(11, NodePairHash);
  int64_t order = 0;
  test_graph.ConstTopoForEachNode([&](const TestNode* node) {
    node_id2order.emplace(node->test_node_id(), order);
    ++order;
  });
  ASSERT_EQ(node_id2order.size(), node_num);

  order = 0;
  test_graph.ConstReverseTopoForEachNode([&](const TestNode* node) {
    node_id2rorder.emplace(node->test_node_id(), order);
    ++order;
  });
  ASSERT_EQ(node_id2rorder.size(), node_num);

  // method : 
  // judge every directed edge <u,v>
  // the node u's order is smaller than v
  int64_t edge_num = 0;
  for (int64_t src_node_id = 0; src_node_id < node_num; ++src_node_id) {
    for (int64_t dst_node_id : graph_conf[src_node_id]) {
      // check topo order
      int64_t src_ord = node_id2order.at(src_node_id);
      int64_t dst_ord = node_id2order.at(dst_node_id);
      ASSERT_LT(src_ord, dst_ord);
      // check reverse-topo order
      src_ord = node_id2rorder.at(src_node_id);
      dst_ord = node_id2rorder.at(dst_node_id);
      ASSERT_GE(src_ord, dst_ord);
      // 
      ++edge_num;
      edges_node_pair.insert(std::make_pair(src_node_id, dst_node_id));
    }
  }

  // 2. judge whether the getter method of Graph can return all nodes and edges
  ASSERT_EQ(test_graph.node_num(), node_num);
  ASSERT_EQ(test_graph.edge_num(), edge_num);
  std::unordered_set<int64_t> node_ids;
  test_graph.ConstForEachNode([&](const TestNode* cur_node) {
    int64_t cur_node_id = cur_node->test_node_id();
    ASSERT_TRUE(node_ids.insert(cur_node_id).second);
    ASSERT_LT(cur_node_id, node_num);
    ASSERT_GE(cur_node_id, 0);
  });
  test_graph.ConstForEachEdge([&](const TestEdge* cur_edge) {
    int64_t src_node_id = cur_edge->src_node()->test_node_id();
    int64_t dst_node_id = cur_edge->dst_node()->test_node_id();
    ASSERT_TRUE(
        edges_node_pair.count(std::make_pair(src_node_id, dst_node_id)) > 0);
  });
}

TEST(TestGraph, test_graph_node_num_7) {
  std::vector<std::vector<int64_t>> graph_conf;
  for (int64_t i = 0; i < 7; ++i) {
    graph_conf.push_back(std::vector<int64_t>());
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
