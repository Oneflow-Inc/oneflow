#ifndef _DAG_DAG_H_
#define _DAG_DAG_H_
#include <glog/logging.h>
#include <cstdint>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <string>
#include <fstream>
#include <iostream>
#include "common/str_util.h"
#include "dag/dag_node.h"
#include "path/base_path.h"
#include "dag/dag_iterator.h"

template <typename DAG, bool isconst = false>
class DagIterator;

template <typename DAG, bool isconst = false>
class DagReverseIterator;

namespace oneflow {
template<typename Data, typename Op>
class Dag {
  friend class DagIterator<Dag<Data, Op>>;
  friend class DagIterator<Dag<Data, Op>, true>;
  friend class DagReverseIterator<Dag<Data, Op>>;
  friend class DagReverseIterator<Dag<Data, Op>, true>;
 public:
  using DNode = DataNode<Data>;
  using ONode = OpNode<Op>;

  Dag();
  Dag(PathType path_type, const std::string& name);
  virtual ~Dag();

  PathType path_type() const;
  std::string name() const;

  template <typename NodeType1, typename NodeType2>
  void AddEdges(NodeType1* node,
    const std::vector<NodeType2*>& inputs,
    const std::vector<NodeType2*>& outputs);

  bool HasOpNode(const std::string& op_name) const;

  DagNode* GetNode(int32_t node_id) const;
  ONode* GetOpNode(int32_t node_id) const;
  ONode* GetOpNode(const std::string& op_name) const;
  DNode* GetDataNode(int32_t node_id) const;
  DNode* GetDataNode(const std::string& data_name) const;

  const DagNode* GetStartNode() const;
  const DagNode* GetEndNode() const;
  bool IsStart(const DagNode* node) const;
  bool IsEnd(const DagNode* node) const;

  // Whether the node is a first (except 'Start') or a last (except 'End')
  // node in the DAG. It is possible that there are multiple FirstOpNode or
  // multiple LastOpNode.
  bool IsFirstOpNode(const DagNode* node) const;
  bool IsFirstOpNode(const std::string& node_name) const;
  bool IsLastOpNode(const DagNode* node) const;
  bool IsLastOpNode(const std::string& node_name) const;
  std::vector<std::string> GetFirstOpNames() const;
  std::vector<std::string> GetLastOpNames() const;

  size_t NumNode() const;
  void PrintDag(const std::string& dag_name,
    bool print_op_name = true,
    bool print_data_name = false);

  int32_t GetNodeLevel(int32_t node_id) const;
  int32_t GetOpNodeLevel(const std::string& op_name) const;
  int32_t GetDataNodeLevel(const std::string& data_name) const;

  std::vector<std::string> GetSucceedingOpNodeNames(
    const std::string& op_name) const;
  std::vector<std::string> GetPrecedingOpNodeNames(
    const std::string& op_name) const;
  std::vector<std::string> GetSucceedingDataNodeNames(
    const std::string& op_name) const;
  std::vector<std::string> GetPrecedingDataNodeNames(
    const std::string& op_name) const;

  std::vector<std::string> GetSucceedingOpNodeNamesOfDataNode(
    const std::string& data_name) const;
  std::vector<std::string> GetPreceedingOpNodeNamesOfDataNode(
    const std::string& data_name) const;
  std::vector<std::string> GetSucceedingDataNodeNamesOfStartNode() const;
  std::vector<std::string> GetPreceedingDataNodeNamesOfEndNode() const;

  std::unordered_set<std::string> GetOpAncestorsOfOpNode(
    const std::string& op_name) const;
  std::unordered_set<std::string> GetOpDescendantsOfOpNode(
    const std::string& op_name) const;

  // Find the data nodes in between the op nodes:|first| and |second|,
  // The returned result may contain zero, one, or more than one data nodes.
  std::vector<std::string> FindDataNodesInBetween(
    const std::string& first,
    const std::string& second) const;
  void RemoveNodeFromDag(int32_t node_id);

 //protected:
  PathType path_type_;
  DagNode* start_;
  int32_t start_id_;
  DagNode* end_;
  int32_t end_id_;
  std::string name_;

  std::unordered_map<int32_t, DagNode*> index_to_node_;
  std::unordered_map<std::string, DNode*> data_name_to_node_;
  std::unordered_map<std::string, ONode*> op_name_to_node_;

  std::unordered_map<int32_t, int32_t> index_to_depth;
  std::map<int32_t, std::unordered_set<int32_t>> depth_to_indices_;
  std::unordered_set<int32_t> first_node_indices_;
  std::unordered_set<int32_t> last_node_indices_;

  std::unordered_map<std::string, std::unordered_set<std::string>>
    op_name_to_ancestors_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
    op_name_to_descendants_;


  DagNode* NewNode(const std::string& name);
  ONode* NewOpNode(const std::string& name);
  DNode* NewDataNode(const std::string& name);

  // Add virtual Start and End node to the DAG. Necessary if we need to use
  // the DagIterator or DagReverseIterator.
  virtual void AddStartAndEndNodes();

  // Calculate the depth of each node in the DAG, and calculate the ancestors
  // and descendants of each node. Necessary if need to perform liveness or 
  // clustering analysis on this DAG.
  void PostProcessing();
  void Clear();

 private:
  int32_t index_counter_;
  void CalculateNodeDepth();
  void MarkFirstAndLastOpNodes();
  void CollectAncestorAndDescendant();

  int32_t NewIndex();

  void PrintEdgesFromNode(
    std::fstream& fs,
    const DagNode* node,
    bool print_op_name,
    bool print_data_name);

  std::string NodeVisualizeShape(const DagNode* node) const;
  std::string NodeVisualizeName(const DagNode* node) const;
  void VisualizeNode(std::fstream& fs,
    int32_t node_id,
    const std::string& node_name,
    NodeType node_type,
    const std::string node_shape,
    bool print_op_name,
    bool print_data_name) const;
  void VisualizeEdge(std::fstream& fs,
    int32_t source_id,
    const std::string& source_name,
    NodeType source_type,
    int32_t sink_id,
    const std::string& sink_name,
    NodeType sink_type,
    bool print_op_name,
    bool print_data_name) const;

  Dag(const Dag& other) = delete;
  Dag& operator=(const Dag& other) = delete;
};

template<typename Data, typename Op>
Dag<Data, Op>::Dag() : index_counter_(0), start_id_(-1), end_id_(-1) {}

template<typename Data, typename Op>
Dag<Data, Op>::Dag(PathType path_type, const std::string& name)
  : path_type_(path_type), index_counter_(0), start_id_(-1), end_id_(-1),
  name_(name) {}

template<typename Data, typename Op>
Dag<Data, Op>::~Dag() {
  Clear();
}

template <typename Data, typename Op>
PathType Dag<Data, Op>::path_type() const {
  return path_type_;
}

template <typename Data, typename Op>
std::string Dag<Data, Op>::name() const {
  return name_;
}

template <typename Data, typename Op>
void Dag<Data, Op>::Clear() {
  std::vector<int32_t> node_ids;
  for (auto i : index_to_node_) {
    node_ids.push_back(i.first);
  }
  for (auto i : node_ids) {
    RemoveNodeFromDag(i);
  }

  index_counter_ = 0;
  start_id_ = -1;
  end_id_ = -1;
  start_ = nullptr;
  end_ = nullptr;

  index_to_node_.clear();
  index_to_depth.clear();
  depth_to_indices_.clear();
  data_name_to_node_.clear();
  op_name_to_node_.clear();
  op_name_to_ancestors_.clear();
  op_name_to_descendants_.clear();
  first_node_indices_.clear();
  last_node_indices_.clear();
}

template<typename Data, typename Op>
inline DagNode* Dag<Data, Op>::GetNode(int32_t node_id) const {
  return index_to_node_.at(node_id);
}

template <typename Data, typename Op>
bool Dag<Data, Op>::IsStart(const DagNode* node) const {
  return node->node_id() == start_id_;
}

template <typename Data, typename Op>
bool Dag<Data, Op>::IsEnd(const DagNode* node) const {
  return node->node_id() == end_id_;
}

template <typename Data, typename Op>
const DagNode* Dag<Data, Op>::GetStartNode() const {
  return start_;
}

template <typename Data, typename Op>
const DagNode* Dag<Data, Op>::GetEndNode() const {
  return end_;
}

template <typename Data, typename Op>
void Dag<Data, Op>::AddStartAndEndNodes() {
  std::vector<DagNode*> start_inputs;
  std::vector<DagNode*> start_outputs;
  std::vector<DagNode*> end_inputs;
  std::vector<DagNode*> end_outputs;

  for (auto& pair : index_to_node_) {
    auto index = pair.first;
    auto node = pair.second;
    // The nodes without predecessors will be all connected as successors of
    // start node.
    if (node->predecessors().size() == 0) {
      start_outputs.push_back(node);
    }
    // The nodes without successors will be all connected as predecessors of
    // end node.
    if (node->successors().size() == 0) {
      end_inputs.push_back(node);
    }
  }

  start_ = NewNode("start");
  start_id_ = start_->node_id();
  end_ = NewNode("end");
  end_id_ = end_->node_id();
  AddEdges(start_, start_inputs, start_outputs);
  AddEdges(end_, end_inputs, end_outputs);
}

template<typename Data, typename Op>
inline typename Dag<Data, Op>::ONode*
Dag<Data, Op>::GetOpNode(int32_t node_id) const {
  auto onode_ptr = dynamic_cast<ONode*>(GetNode(node_id));
  CHECK_NOTNULL(onode_ptr);
  CHECK(onode_ptr->Type() == NodeType::kOpNode);
  return onode_ptr;
}

template<typename Data, typename Op>
inline typename Dag<Data, Op>::DNode*
Dag<Data, Op>::GetDataNode(int32_t node_id) const {
  auto dnode_ptr = dynamic_cast<DNode*>(GetNode(node_id));
  CHECK_NOTNULL(dnode_ptr);
  CHECK(dnode_ptr->Type() == NodeType::kDataNode);
  return dnode_ptr;
}

template <typename Data, typename Op>
DataNode<Data>* Dag<Data, Op>::GetDataNode(const std::string& data_name) const {
  auto it = data_name_to_node_.find(data_name);
  CHECK(it != data_name_to_node_.end());
  return it->second;
}

template <typename Data, typename Op>
OpNode<Op>* Dag<Data, Op>::GetOpNode(const std::string& op_name) const {
  auto it = op_name_to_node_.find(op_name);
  CHECK(it != op_name_to_node_.end())
    << "Can not find op_name: " << op_name;
  return it->second;
}

template <typename Data, typename Op>
bool Dag<Data, Op>::HasOpNode(const std::string& op_name) const {
  auto it = op_name_to_node_.find(op_name);
  return it != op_name_to_node_.end();
}

template<typename Data, typename Op>
inline size_t Dag<Data, Op>::NumNode() const {
  return index_to_node_.size();
}

template<typename Data, typename Op>
inline int32_t Dag<Data, Op>::NewIndex() {
  return index_counter_++;
}

template<typename Data, typename Op>
typename Dag<Data, Op>::DNode* Dag<Data, Op>::NewDataNode(
  const std::string& name) {
  auto ret = new DNode(NewIndex(), name);
  index_to_node_.insert(std::pair<int32_t, DagNode*>(ret->node_id(), ret));
  return ret;
}

template<typename Data, typename Op>
typename Dag<Data, Op>::ONode* Dag<Data, Op>::NewOpNode(
  const std::string& name) {
  auto ret = new ONode(NewIndex(), name);
  index_to_node_.insert(std::pair<int32_t, DagNode*>(ret->node_id(), ret));
  return ret;
}

template<typename Data, typename Op>
DagNode* Dag<Data, Op>::NewNode(const std::string& name) {
  auto ret = new DagNode(NewIndex(), name);
  index_to_node_.insert(std::pair<int32_t, DagNode*>(ret->node_id(), ret));
  return ret;
}

template<typename Data, typename Op>
template <typename NodeType1, typename NodeType2>
void Dag<Data, Op>::AddEdges(NodeType1* node,
    const std::vector<NodeType2*>& inputs,
    const std::vector<NodeType2*>& outputs) {
  for (auto& input : inputs) {
    node->AddParent(input);
  }
  for (auto& output : outputs) {
    output->AddParent(node);
  }
}

template <typename Data, typename Op>
void Dag<Data, Op>::PrintDag(const std::string& dag_name,
  bool print_op_name, bool print_data_name) {
  std::fstream local_fs;
  std::string image_name = "./visualize/" + dag_name + ".dot";
  local_fs.open(image_name, std::fstream::out);
  local_fs << "digraph " << dag_name << " {" << std::endl;
  local_fs << "rankdir=TB;" << std::endl;
  std::vector<DNode*> inputs;
  std::vector<DNode*> outputs;

  DagIterator<Dag<Data, Op>, true> dag_iterator(*this);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    PrintEdgesFromNode(local_fs, current_node, print_op_name, print_data_name);
  }
  local_fs << "}" << std::endl;
  local_fs.close();
}

template <typename Data, typename Op>
std::string Dag<Data, Op>::NodeVisualizeShape(const DagNode* node) const {
  std::string shape;
  if (node->Type() == NodeType::kUnknown) {
    shape = "diamond";
  } else if (node->Type() == NodeType::kOpNode) {
    shape = "box";
  } else if (node->Type() == NodeType::kDataNode) {
    shape = "circle";
  }
  return shape;
}

template <typename Data, typename Op>
std::string Dag<Data, Op>::NodeVisualizeName(const DagNode* node) const {
  std::string name;
  if (node->Type() == NodeType::kUnknown) {
    name = node->node_name();
  } else if (node->Type() == NodeType::kOpNode) {
    name = node->node_name();
  } else if (node->Type() == NodeType::kDataNode) {
    name = node->node_name();
  }
  return name;
}

template <typename Data, typename Op>
void Dag<Data, Op>::VisualizeNode(
  std::fstream& fs,
  int32_t node_id,
  const std::string& node_name,
  NodeType node_type,
  const std::string node_shape,
  bool print_op_name,
  bool print_data_name) const {
  if (node_type == NodeType::kDataNode) {
    if (print_data_name) {
      fs << "\"" << node_name;
    } else {
      fs << "\"" << node_id;
    }
    fs << "\" [shape=" << node_shape << "];" << std::endl;
  } else {
    if (print_op_name) {
      fs << "\"" << node_name;
    } else {
      fs << "\"" << node_id;
    }
    fs << "\" [shape=" << node_shape << "];" << std::endl;
  }
}

template <typename Data, typename Op>
void Dag<Data, Op>::VisualizeEdge(
  std::fstream& fs,
  int32_t source_id,
  const std::string& source_name,
  NodeType source_type,
  int32_t sink_id,
  const std::string& sink_name,
  NodeType sink_type,
  bool print_op_name,
  bool print_data_name) const {
  if (source_type == NodeType::kDataNode) {
    if (print_data_name) {
      fs << "\"" << source_name << "\"";
    } else {
      fs << "\"" << source_id << "\"";
    }
  } else {
    if (print_op_name) {
      fs << "\"" << source_name << "\"";
    } else {
      fs << "\"" << source_id << "\"";
    }
  }
  fs << " -> ";
  if (sink_type == NodeType::kDataNode) {
    if (print_data_name) {
      fs << "\"" << sink_name << "\"" << std::endl;
    } else {
      fs << "\"" << sink_id << "\"" << std::endl;
    }
  } else {
    if (print_op_name) {
      fs << "\"" << sink_name << "\"" << std::endl;
    } else {
      fs << "\"" << sink_id << "\"" << std::endl;
    }
  }
}

template <typename Data, typename Op>
void Dag<Data, Op>::PrintEdgesFromNode(std::fstream& fs,
  const DagNode* node, bool print_op_name, bool print_data_name) {
  int32_t node_id = node->node_id();
  NodeType node_type = node->Type();
  auto node_shape = NodeVisualizeShape(node);
  auto node_name = NodeVisualizeName(node);
  VisualizeNode(fs, node_id, node_name, node_type, node_shape, print_op_name,
    print_data_name);

  auto successors = node->successors();
  for (auto& successor_id : successors) {
    auto successor = GetNode(successor_id);
    NodeType successor_type = successor->Type();
    auto successor_shape = NodeVisualizeShape(successor);
    auto successor_name = NodeVisualizeName(successor);
    VisualizeNode(fs, successor_id, successor_name,
      successor_type, successor_shape, print_op_name, print_data_name);
    VisualizeEdge(fs, node_id, node_name, node_type, successor_id,
      successor_name, successor_type, print_op_name, print_data_name);
  }
}

template<typename Data, typename Op>
void Dag<Data, Op>::RemoveNodeFromDag(int32_t node_id) {
  auto node = GetNode(node_id);
  for (auto successor_id : node->mutable_successors()) {
    auto successor = GetNode(successor_id);
    CHECK_EQ(successor->mutable_predecessors().erase(node_id), 1);
  }
  for (auto predecessor_id : node->mutable_predecessors()) {
    auto predecessor = GetNode(predecessor_id);
    CHECK_EQ(predecessor->mutable_successors().erase(node_id), 1);
  }
  CHECK_EQ(index_to_node_.erase(node_id), 1);
  delete node;
  return;
}

template <typename Data, typename Op>
int32_t Dag<Data, Op>::GetOpNodeLevel(const std::string& op_name) const {
  auto it = op_name_to_node_.find(op_name);
  CHECK(it != op_name_to_node_.end());
  auto op_node = it->second;
  auto op_node_id = op_node->node_id();
  return GetNodeLevel(op_node_id);
}

template <typename Data, typename Op>
int32_t Dag<Data, Op>::GetDataNodeLevel(const std::string& data_name) const {
  auto it = data_name_to_node_.find(data_name);
  CHECK(it != data_name_to_node_.end());
  auto data_node = it->second;
  auto data_node_id = data_node->node_id();
  return GetNodeLevel(data_node_id);
}

template <typename Data, typename Op>
int32_t Dag<Data, Op>::GetNodeLevel(int32_t node_id) const {
  auto it = index_to_depth.find(node_id);
  CHECK(it != index_to_depth.end());
  return it->second;
}

template <typename Data, typename Op>
void Dag<Data, Op>::CalculateNodeDepth() {
  index_to_depth.clear();
  depth_to_indices_.clear();
  DagIterator<Dag<Data, Op>, true> dag_iterator(*this);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    auto current_id = current_node->node_id();
    auto& predecessors = current_node->predecessors();
    auto& successors = current_node->successors();
    if (predecessors.size() == 0) {
      // Set the depth of the start id to 0
      index_to_depth.insert(std::make_pair(current_id, 0));
      std::unordered_set<int32_t> s_cid;
      s_cid.insert(current_id);
      depth_to_indices_.insert(std::make_pair(0, s_cid));
    } else {
      // Set the depth of current_node according to its deepest predecessor
      int32_t level = -1;
      for (auto&& predecessor_id : predecessors) {
        // auto predecessor_id = predecessor->node_id();
        CHECK(index_to_depth.count(predecessor_id) > 0);
        auto predecessor_level = index_to_depth[predecessor_id];
        if (predecessor_level > level) {
          level = predecessor_level;
        }
      }
      CHECK(level != -1);
      ++level;
      index_to_depth.insert(std::make_pair(current_id, level));
      auto it = depth_to_indices_.find(level);
      if (it == depth_to_indices_.end()) {
        std::unordered_set<int32_t> s_cid;
        s_cid.insert(current_id);
        depth_to_indices_.insert(std::make_pair(level, s_cid));
      } else {
        it->second.insert(current_id);
      }
    }
  }
}

template <typename Data, typename Op>
void Dag<Data, Op>::CollectAncestorAndDescendant() {
  op_name_to_ancestors_.clear();
  DagIterator<Dag<Data, Op>, true> dag_iterator(*this);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    auto current_id = current_node->node_id();
    auto current_name = current_node->node_name();
    if (current_node->Type() != NodeType::kOpNode) continue;
    std::unordered_set<std::string> ancestors_of_current_node;
    auto predecessors = GetPrecedingOpNodeNames(current_name);
    for (auto&& predecessor : predecessors) {
      auto ancestor_it = op_name_to_ancestors_.find(predecessor);
      if (ancestor_it != op_name_to_ancestors_.end()) {
        auto ancestors_of_predecessor = ancestor_it->second;
        // Add the predecessor's ancestors
        for (auto& ancestor_of_predecessor : ancestors_of_predecessor) {
          ancestors_of_current_node.insert(ancestor_of_predecessor);
        }
      }
      // Add the predecessor
      ancestors_of_current_node.insert(predecessor);
    }
    op_name_to_ancestors_.insert(std::make_pair(current_name, ancestors_of_current_node));
  }

  op_name_to_descendants_.clear();
  DagReverseIterator<Dag<Data, Op>, true> dag_riterator(*this);
  for (dag_riterator.First(); !dag_riterator.IsDone(); dag_riterator.Next()) {
    auto current_node = dag_riterator.CurrentNode();
    auto current_id = current_node->node_id();
    auto current_name = current_node->node_name();
    if (current_node->Type() != NodeType::kOpNode) continue;
    std::unordered_set<std::string> descendants_of_current_node;
    auto successors = GetSucceedingOpNodeNames(current_name);
    for (auto successor : successors) {
      auto descendant_it = op_name_to_descendants_.find(successor);
      if (descendant_it != op_name_to_descendants_.end()) {
        auto descendants_of_successor = descendant_it->second;
        // Add the successor's descendants
        for (auto&& descendant_of_successor : descendants_of_successor) {
          descendants_of_current_node.insert(descendant_of_successor);
        }
      }
      // Add the successor
      descendants_of_current_node.insert(successor);
    }
    op_name_to_descendants_.insert(std::make_pair(current_name, descendants_of_current_node));
  }
}

template <typename Data, typename Op>
std::unordered_set<std::string> Dag<Data, Op>::GetOpAncestorsOfOpNode(
  const std::string& op_name) const {
  auto ancestor_it = op_name_to_ancestors_.find(op_name);
  CHECK(ancestor_it != op_name_to_ancestors_.end());
  return ancestor_it->second;
}

template <typename Data, typename Op>
std::unordered_set<std::string> Dag<Data, Op>::GetOpDescendantsOfOpNode(
  const std::string& op_name) const {
  auto descendant_it = op_name_to_descendants_.find(op_name);
  CHECK(descendant_it != op_name_to_descendants_.end());
  return descendant_it->second;
}

template <typename Data, typename Op>
std::vector<std::string> Dag<Data, Op>::FindDataNodesInBetween(
  const std::string& first,
  const std::string& second) const {
  std::vector<std::string> data_names;

  ONode* first_node = GetOpNode(first);
  auto& successors = first_node->successors();
  ONode* second_node = GetOpNode(second);
  auto& predecessors = second_node->predecessors();

  for (auto& successor_id : successors) {
    auto successor_node = GetNode(successor_id);
    auto& successor_name = successor_node->node_name();
    for (auto& predecessor_id : predecessors) {
      auto predecessor_node = GetNode(predecessor_id);
      auto& predecessor_name = predecessor_node->node_name();
      if (successor_name == predecessor_name) {
        data_names.push_back(successor_name);
      }
    }
  }
  // CHECK(data_names.size() != 0) << "No middle data blob found";
  return data_names;
}

template <typename Data, typename Op>
void Dag<Data, Op>::PostProcessing() {
  CalculateNodeDepth();
  MarkFirstAndLastOpNodes();
  CollectAncestorAndDescendant();
}

template <typename Data, typename Op>
void Dag<Data, Op>::MarkFirstAndLastOpNodes() {
  // Get the levels of first and last op nodes
  int32_t depth_first = std::numeric_limits<int32_t>::max();
  int32_t depth_last = -1;
  for (auto& depth_index_pair : depth_to_indices_) {
    int32_t depth = depth_index_pair.first;
    auto indices = depth_index_pair.second;
    for (int32_t index : indices) {
      auto node = GetNode(index);
      if (node->Type() == NodeType::kOpNode) {
        if (depth_first > depth) {
          depth_first = depth;
        }
        if (depth_last < depth) {
          depth_last = depth;
        }
      }
    }
  }
  first_node_indices_ = depth_to_indices_[depth_first];
  last_node_indices_ = depth_to_indices_[depth_last];
}

template <typename Data, typename Op>
std::vector<std::string> Dag<Data, Op>::GetSucceedingOpNodeNames(
  const std::string& op_name) const {
  std::vector<std::string> succeeding_op_ids;
  std::unordered_set<std::string> succeeding_op_set;
  ONode* onode = GetOpNode(op_name);
  auto& successors = onode->successors();
  for (auto& data_node_id : successors) {
    auto data_node = GetNode(data_node_id);
    if (data_node->Type() != NodeType::kDataNode) continue;
    auto& data_node_successors = data_node->successors();
    for (auto& op_node_id : data_node_successors) {
      auto op_node = GetNode(op_node_id);
      if (op_node->Type() != NodeType::kOpNode) continue;
      // There may be multiple paths from an op node to its succeeding op node
      if (succeeding_op_set.count(op_node->node_name()) > 0) continue;
      succeeding_op_set.insert(op_node->node_name());
      succeeding_op_ids.push_back(op_node->node_name());
    }
  }
  return succeeding_op_ids;
}

template <typename Data, typename Op>
std::vector<std::string> Dag<Data, Op>::GetSucceedingOpNodeNamesOfDataNode(
  const std::string& data_name) const {
  std::vector<std::string> succeeding_op_names;
  DNode* data_node = GetDataNode(data_name);
  auto& data_node_successors = data_node->successors();
  for (auto& op_node_id : data_node_successors) {
    auto op_node = GetNode(op_node_id);
    if (op_node->Type() != NodeType::kOpNode) continue;
    succeeding_op_names.push_back(op_node->node_name());
  }
  return succeeding_op_names;
}

template <typename Data, typename Op>
std::vector<std::string> Dag<Data, Op>::GetPreceedingOpNodeNamesOfDataNode(
  const std::string& data_name) const {
  std::vector<std::string> preceeding_op_names;
  DNode* data_node = GetDataNode(data_name);
  auto& data_node_predecessors = data_node->predecessors();
  for (auto& op_node_id : data_node_predecessors) {
    auto op_node = GetNode(op_node_id);
    if (op_node->Type() != NodeType::kOpNode) continue;
    preceeding_op_names.push_back(op_node->node_name());
  }
  return preceeding_op_names;
}

template <typename Data, typename Op>
std::vector<std::string> Dag<Data, Op>::GetSucceedingDataNodeNamesOfStartNode()
  const {
  std::vector<std::string> succeeding_data_names;
  auto& data_node_successors = start_->successors();
  for (auto& data_node_id : data_node_successors) {
    auto data_node = GetNode(data_node_id);
    // NOTE(jiyuan): in some DAG, the successors of START node is op node. Only
    // in some particular DAGs, the successors of START node is data node, such
    // as TaskDag
    CHECK(data_node->Type() == NodeType::kDataNode);
    succeeding_data_names.push_back(data_node->node_name());
  }
  return succeeding_data_names;
}

template <typename Data, typename Op>
std::vector<std::string> Dag<Data, Op>::GetPreceedingDataNodeNamesOfEndNode()
  const {
  std::vector<std::string> preceeding_data_names;
  auto& data_node_predecessors = end_->predecessors();
  for (auto& data_node_id : data_node_predecessors) {
    auto data_node = GetDataNode(data_node_id);
    CHECK(data_node->Type() == NodeType::kDataNode);
    preceeding_data_names.push_back(data_node->node_name());
  }
  return preceeding_data_names;
}

template <typename Data, typename Op>
std::vector<std::string> Dag<Data, Op>::GetPrecedingOpNodeNames(
  const std::string& op_name) const {
  std::vector<std::string> preceding_op_ids;
  std::unordered_set<std::string> preceding_op_set;
  ONode* onode = GetOpNode(op_name);
  auto& predecessors = onode->predecessors();
  for (auto node_id : predecessors) {
    auto node = GetNode(node_id);
    if (node->Type() != NodeType::kDataNode) continue;
    auto node_predecessors = node->predecessors();
    for (auto& op_node_id : node_predecessors) {
      auto op_node = GetNode(op_node_id);
      if (op_node->Type() != NodeType::kOpNode) continue;
      if (preceding_op_set.count(op_node->node_name()) > 0) continue;
      preceding_op_set.insert(op_node->node_name());
      preceding_op_ids.push_back(op_node->node_name());
    }
  }
  return preceding_op_ids;
}

template <typename Data, typename Op>
std::vector<std::string> Dag<Data, Op>::GetSucceedingDataNodeNames(
  const std::string& op_name) const {
  std::vector<std::string> succeeding_data_ids;
  ONode* onode = GetOpNode(op_name);
  auto& successors = onode->successors();
  for (auto& data_node_id : successors) {
    auto data_node = GetNode(data_node_id);
    if (data_node->Type() != NodeType::kDataNode) continue;
    succeeding_data_ids.push_back(data_node->node_name());
  }
  return succeeding_data_ids;
}

template <typename Data, typename Op>
std::vector<std::string> Dag<Data, Op>::GetPrecedingDataNodeNames(
  const std::string& op_name) const {
  std::vector<std::string> preceding_data_ids;
  ONode* onode = GetOpNode(op_name);
  auto& predecessors = onode->predecessors();
  for (auto& data_node_id : predecessors) {
    auto data_node = GetNode(data_node_id);
    if (data_node->Type() != NodeType::kDataNode) continue;
    preceding_data_ids.push_back(data_node->node_name());
  }
  return preceding_data_ids;
}

template <typename Data, typename Op>
bool Dag<Data, Op>::IsFirstOpNode(const DagNode* node) const {
  auto node_id = node->node_id();
  auto it = first_node_indices_.find(node_id);
  return it != first_node_indices_.end();
}

template <typename Data, typename Op>
bool Dag<Data, Op>::IsFirstOpNode(const std::string& node_name) const {
  auto node = GetOpNode(node_name);
  return IsFirstOpNode(node);
}

template <typename Data, typename Op>
bool Dag<Data, Op>::IsLastOpNode(const DagNode* node) const {
  auto node_id = node->node_id();
  auto it = last_node_indices_.find(node_id);
  return it != last_node_indices_.end();
}

template <typename Data, typename Op>
bool Dag<Data, Op>::IsLastOpNode(const std::string& node_name) const {
  auto node = GetOpNode(node_name);
  return IsLastOpNode(node);
}

template <typename Data, typename Op>
std::vector<std::string> Dag<Data, Op>::GetFirstOpNames() const {
  std::vector<std::string> op_names;
  for (auto index : first_node_indices_) {
    auto it = index_to_node_.find(index);
    CHECK(it != index_to_node_.end());
    op_names.push_back(it->second->node_name());
  }
  return op_names;
}

template <typename Data, typename Op>
std::vector<std::string> Dag<Data, Op>::GetLastOpNames() const {
  std::vector<std::string> op_names;
  for (auto index : last_node_indices_) {
    auto it = index_to_node_.find(index);
    CHECK(it != index_to_node_.end());
    op_names.push_back(it->second->node_name());
  }
  return op_names;
}

}  // namespace oneflow
#endif  // _DAG_DAG_H_
