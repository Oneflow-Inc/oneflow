#include "dag/dag_node.h"
#include <vector>
#include "gtest/gtest.h"
#include "common/util.h"

namespace oneflow {

TEST(DagNode, node_id_is_unique) {
  for (int32_t i = 0; i < 100; ++i) {
    DataNode<int32_t> new_node;
    new_node.init("new_node", std::make_shared<int> (0));
    ASSERT_EQ(new_node.node_id(), i);
  }
}

TEST(DagNode, add_remove_predecessor) {
  std::vector<DataNode<size_t>> node_vec(4);
  for (size_t i = 0; i < node_vec.size(); ++i) {
    node_vec[i].init(std::to_string(i), std::make_shared<size_t> (i));
  }

  // add: 0->1;0->2;1->3;2->3
  ASSERT_EQ(true, node_vec[1].AddPredecessor(&node_vec[0]));
  ASSERT_EQ(true, node_vec[2].AddPredecessor(&node_vec[0]));
  ASSERT_EQ(true, node_vec[3].AddPredecessor(&node_vec[1]));
  ASSERT_EQ(true, node_vec[3].AddPredecessor(&node_vec[2]));
  
  // check successors and predecessors
  ASSERT_TRUE(IsEqual(node_vec[0].successors(), {node_vec[1].node_id(),
                                                  node_vec[2].node_id()}));
  ASSERT_TRUE(IsEqual(node_vec[1].successors(), {node_vec[3].node_id()}));
  ASSERT_TRUE(IsEqual(node_vec[2].successors(), {node_vec[3].node_id()}));
  ASSERT_TRUE(IsEqual(node_vec[3].successors(), {}));
  
  ASSERT_TRUE(IsEqual(node_vec[0].predecessors(), {}));
  ASSERT_TRUE(IsEqual(node_vec[1].predecessors(), {node_vec[0].node_id()}));
  ASSERT_TRUE(IsEqual(node_vec[2].predecessors(), {node_vec[0].node_id()}));
  ASSERT_TRUE(IsEqual(node_vec[3].predecessors(), {node_vec[1].node_id(),
                                                    node_vec[2].node_id()}));
  
  // false add
  ASSERT_EQ(false, node_vec[1].AddPredecessor(&node_vec[0]));
  ASSERT_EQ(false, node_vec[2].AddPredecessor(&node_vec[0]));
  ASSERT_EQ(false, node_vec[3].AddPredecessor(&node_vec[1]));
  ASSERT_EQ(false, node_vec[3].AddPredecessor(&node_vec[2]));
  
  // true remove
  ASSERT_EQ(true, node_vec[1].RemovePredecessor(&node_vec[0]));
  ASSERT_EQ(true, node_vec[3].RemovePredecessor(&node_vec[2]));
  
  // check successors and predecessors
  ASSERT_TRUE(IsEqual(node_vec[0].successors(), {node_vec[2].node_id()}));
  ASSERT_TRUE(IsEqual(node_vec[1].successors(), {node_vec[3].node_id()}));
  ASSERT_TRUE(IsEqual(node_vec[2].successors(), {}));
  ASSERT_TRUE(IsEqual(node_vec[3].successors(), {}));
  
  ASSERT_TRUE(IsEqual(node_vec[0].predecessors(), {}));
  ASSERT_TRUE(IsEqual(node_vec[1].predecessors(), {}));
  ASSERT_TRUE(IsEqual(node_vec[2].predecessors(), {node_vec[0].node_id()}));
  ASSERT_TRUE(IsEqual(node_vec[3].predecessors(), {node_vec[1].node_id()}));
  
  // false remove
  ASSERT_EQ(false, node_vec[1].RemovePredecessor(&node_vec[0]));
  ASSERT_EQ(false, node_vec[3].RemovePredecessor(&node_vec[2]));
}

} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
