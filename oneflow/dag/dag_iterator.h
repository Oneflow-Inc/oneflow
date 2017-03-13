#ifndef _DAG_DAG_ITERATOR_H_
#define _DAG_DAG_ITERATOR_H_
#include <unordered_map>
#include <queue>
namespace caffe {

class DagNode;

template <bool flag, typename IsTrue, typename IsFalse>
struct choose;
template <typename IsTrue, typename IsFalse>
struct choose<true, IsTrue, IsFalse> {
  using type = IsTrue;
};
template <typename IsTrue, typename IsFalse>
struct choose<false, IsTrue, IsFalse> {
  using type = IsFalse;
};

template <typename DAG, bool isconst = false>
class DagIterator {
  public:
    using dag_reference = typename choose<isconst, const DAG&, DAG&>::type;
    using node_pointer = typename choose<isconst, const DagNode*, DagNode*>::type;
  public:
    explicit DagIterator(dag_reference dag);
    ~DagIterator();

    void First();
    bool IsDone();
    void Next();
    node_pointer CurrentNode();
  private:
    dag_reference dag_;
    std::unordered_map<int32_t, int32_t> in_edge_map_;
    std::queue<int32_t> node_id_queue_;
    node_pointer current_node_;

    void CountInEdges();
    void ProcessCurrentNodeSuccessors();

    DagIterator(const DagIterator& other) = delete;
    DagIterator& operator=(const DagIterator&other) = delete;
};
template <typename DAG, bool isconst = false>
class DagReverseIterator {
  public:
    using dag_reference = typename choose<isconst, const DAG&, DAG&>::type;
    using node_pointer = typename choose<isconst, const DagNode*, DagNode*>::type;
  public:
    explicit DagReverseIterator(dag_reference dag);
    ~DagReverseIterator();

    void First();
    bool IsDone();
    void Next();
    node_pointer CurrentNode();
  private:
    dag_reference dag_;
    std::unordered_map<int32_t, int32_t> in_edge_map_;
    std::queue<int32_t> node_id_queue_;
    node_pointer current_node_;

    void CountInEdges();
    void ProcessCurrentNodePredecessors();

    DagReverseIterator(const DagReverseIterator& other) = delete;
    DagReverseIterator& operator=(const DagReverseIterator&other) = delete;
};

template <typename DAG, bool isconst>
DagIterator<DAG, isconst>::DagIterator(dag_reference dag) : dag_(dag),
  current_node_(nullptr), in_edge_map_(), node_id_queue_() {
}
template <typename DAG, bool isconst>
DagIterator<DAG, isconst>::~DagIterator() {}
template <typename DAG, bool isconst>
void DagIterator<DAG, isconst>::First() {
  in_edge_map_.clear();
  while (!node_id_queue_.empty()) node_id_queue_.pop();
  CountInEdges();
  CHECK(dag_.start_id_ != -1) << "Make sure to set the start_id_";
  current_node_ = dag_.GetNode(dag_.start_id_);
  ProcessCurrentNodeSuccessors();
  return;
}

template <typename DAG, bool isconst>
bool DagIterator<DAG, isconst>::IsDone() {
  return current_node_ == NULL;
}

template <typename DAG, bool isconst>
void DagIterator<DAG, isconst>::Next() {
  if (node_id_queue_.empty()) {
    current_node_ = NULL;
  }
  else {
    int32_t current_id = node_id_queue_.front(); node_id_queue_.pop();
    current_node_ = dag_.GetNode(current_id);
    ProcessCurrentNodeSuccessors();
  }
}

template <typename DAG, bool isconst>
typename choose<isconst, const DagNode*, DagNode*>::type
  DagIterator<DAG, isconst>::CurrentNode() {
  return current_node_;
}

template <typename DAG, bool isconst>
void DagIterator<DAG, isconst>::CountInEdges() {
  for (auto& node_pair : dag_.index_to_node_) {
    int32_t node_id = node_pair.first;
    auto node = node_pair.second;
    int32_t predecessor_num = node->predecessors().size();
    in_edge_map_.insert({node_id, predecessor_num});
  }
}

template <typename DAG, bool isconst>
void DagIterator<DAG, isconst>::ProcessCurrentNodeSuccessors() {
  auto& successors = current_node_->successors();
  for (auto next_id : successors) {
    CHECK_EQ(in_edge_map_.count(next_id), 1);
    in_edge_map_[next_id]--;
    if (in_edge_map_[next_id] ==  0) {
      node_id_queue_.push(next_id);
    }
  }
}

template <typename DAG, bool isconst>
DagReverseIterator<DAG, isconst>::DagReverseIterator(dag_reference dag) : dag_(dag),
  current_node_(nullptr), in_edge_map_(), node_id_queue_() {
}

template <typename DAG, bool isconst>
DagReverseIterator<DAG, isconst>::~DagReverseIterator() {}
template <typename DAG, bool isconst>
void DagReverseIterator<DAG, isconst>::First() {
  in_edge_map_.clear();
  while (!node_id_queue_.empty()) node_id_queue_.pop();
  CountInEdges();
  CHECK(dag_.end_id_ != -1) << "Make sure to set the end_id_";
  current_node_ = dag_.GetNode(dag_.end_id_);
  ProcessCurrentNodePredecessors();
  return;
}

template <typename DAG, bool isconst>
bool DagReverseIterator<DAG, isconst>::IsDone() {
  return current_node_ == NULL;
}

template <typename DAG, bool isconst>
void DagReverseIterator<DAG, isconst>::Next() {
  if (node_id_queue_.empty()) {
    current_node_ = NULL;
  }
  else {
    int32_t current_id = node_id_queue_.front(); node_id_queue_.pop();
    current_node_ = dag_.GetNode(current_id);
    ProcessCurrentNodePredecessors();
  }
}

template <typename DAG, bool isconst>
typename choose<isconst, const DagNode*, DagNode*>::type
  DagReverseIterator<DAG, isconst>::CurrentNode() {
  return current_node_;
}

template <typename DAG, bool isconst>
void DagReverseIterator<DAG, isconst>::CountInEdges() {
  for (auto& node_pair : dag_.index_to_node_) {
    int32_t node_id = node_pair.first;
    auto node = node_pair.second;
    int32_t successor_num = node->successors().size();
    in_edge_map_.insert({node_id, successor_num});
  }
}

template <typename DAG, bool isconst>
void DagReverseIterator<DAG, isconst>::ProcessCurrentNodePredecessors() {
  auto& predecessors = current_node_->predecessors();
  for (auto& next_id : predecessors) {
    CHECK_EQ(in_edge_map_.count(next_id), 1);
    in_edge_map_[next_id]--;
    if (in_edge_map_[next_id] ==  0) {
      node_id_queue_.push(next_id);
    }
  }
}
}  // namespace caffe
#endif  // _DAG_DAG_ITERATOR_H_