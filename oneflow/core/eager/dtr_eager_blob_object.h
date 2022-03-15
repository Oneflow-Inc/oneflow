#ifndef ONEFLOW_CORE_EAGER_DTR_EAGER_BLOB_OBJECT_H_
#define ONEFLOW_CORE_EAGER_DTR_EAGER_BLOB_OBJECT_H_

#include "oneflow/core/eager/eager_blob_object.h"

namespace oneflow {

namespace vm {

class LocalCallOpKernelPhyInstrOperand;
class DTRInstrOperand;

class DisjNode {
 public:
  DisjNode(double time) : compute_time_(time), parent_(nullptr), pesudo_node_(nullptr), cnt_(1) {}

  bool is_root() { return !bool(parent_); }

  void set_parent(std::shared_ptr<DisjNode>& parent) { parent_ = parent; }
  void set_pesudo_node(std::shared_ptr<DisjNode>& pesudo_node) { pesudo_node_ = pesudo_node; }
  void set_compute_time(double new_time) {
    compute_time_ = new_time;
  }

  void set_cnt(int cnt) { cnt_ = cnt; }
  void add_cnt() { cnt_++; }
  void reduce_cnt() { cnt_--; }

  double compute_time() { return compute_time_; }
  std::shared_ptr<DisjNode> parent() { return parent_; }
  std::shared_ptr<DisjNode> pesudo_node() { return pesudo_node_; }
  int cnt() { return cnt_; }

  void reset(double t) {
    compute_time_ = t;
    parent_.reset();
  }
  void reset_pesudo_node();

 private:
  double compute_time_;
  std::shared_ptr<DisjNode> parent_;
  std::shared_ptr<DisjNode> pesudo_node_;
  int cnt_;
};

class DTREagerBlobObject final : public EagerBlobObject {
 public:
  DTREagerBlobObject(const DTREagerBlobObject&) = delete;
  DTREagerBlobObject(DTREagerBlobObject&&) = delete;
  DTREagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case,
                     const std::shared_ptr<Shape>& shape, DataType data_type,
                     const std::shared_ptr<TensorBuffer>& tensor_buffer)
      : DTREagerBlobObject(mem_case, shape, data_type, tensor_buffer, nullptr) {}
  DTREagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case,
                     const std::shared_ptr<Shape>& shape, DataType data_type,
                     const std::shared_ptr<TensorBuffer>& tensor_buffer, LocalDepObject* dep_object);
  ~DTREagerBlobObject() override;

  Maybe<void> TryAllocateBlobBodyMemory(DeviceCtx* device_ctx) override;

  char* object_dptr() { return tensor_buffer_->blob_dptr(); }

  double blob_body_bytes_double() const { return static_cast<double>(blob_body_bytes_); }

  void set_compute_op(const std::shared_ptr<LocalCallOpKernelPhyInstrOperand>& operand);

  int parent_depth() const;
  int child_depth() const;

  // Getters and Setters
  const std::size_t memory() const { return blob_body_bytes_; }
  const double compute_time() const { return compute_time_; }
  const double last_access_time() const { return last_access_time_; }
  DTRInstrOperand* compute_op() const { return compute_op_.get(); }
  Maybe<DTRInstrOperand*> user_op(int i) const {
    CHECK_LT_OR_RETURN(i, user_ops_.size());
    CHECK_NOTNULL_OR_RETURN(user_ops_[i].get());
    return user_ops_[i].get();
  }
  void set_compute_time(double val);
  void set_last_access_time(double val) { last_access_time_ = val; }
  void set_evict_attr(bool val) { could_evict_ = val; }
  void set_bp_required(bool val) { is_bp_required_ = val; }
  void set_recompute_mode(int val) const { recompute_mode_ = val; }

  const std::string& compute_op_type_name() const;

  // DTR Strategy
  bool is_in_memory() const;
  bool is_pinned() const { return (pinned_ > 0); }
  int num_pinned() const { return pinned_; }
  int num_user_ops() const { return user_ops_.size(); }
  bool is_evictable() const;
  bool is_bp_required() const { return is_bp_required_; }

  void pin();
  void unpin();
  void update_access_time();
  void AppendUserOp(const std::shared_ptr<LocalCallOpKernelPhyInstrOperand>& operand);
  Maybe<void> evict();
  Maybe<double> parent_cost(bool is_bp_required=false) const;
  Maybe<double> child_cost(bool is_bp_required=false) const;
  Maybe<double> neighbor_cost() const;
  Maybe<double> approx_neighbor_cost() const;
  Maybe<double> rev_fwd_cost() const;
  Maybe<double> rev_bwd_cost() const;
  size_t input_size() const;
  void clear_invalid_object();

  // TODO: variable cost functions in terms of different heuristics
  Maybe<double> cost() const;
  Maybe<double> cost(const std::string& heuristic) const;
  Maybe<double> reverse_cost() const;

  std::shared_ptr<DisjNode> node;
  void reset_node(double t) {
    node->reset(t);
  }

  void reset_pesudo_node();

  const int pesudo_cnt() const {
    auto&& pesudo_ = node->pesudo_node();
    int cnt = pesudo_->cnt();
    return cnt;
  }

 private:
  bool evict_flag_ = false;
  bool could_evict_;
  bool is_bp_required_;
  double compute_time_;
  double last_access_time_;
  size_t pinned_;
  mutable int recompute_mode_;    // 1 - forward recomputation; 0-1 - reverse recomputation
  std::unique_ptr<DTRInstrOperand> compute_op_;
  std::vector<std::unique_ptr<DTRInstrOperand>> user_ops_;
};

}
}

#endif
