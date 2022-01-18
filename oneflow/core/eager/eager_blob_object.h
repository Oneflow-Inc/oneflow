/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_EAGER_EAGER_BLOB_OBJECT_H_
#define ONEFLOW_CORE_EAGER_EAGER_BLOB_OBJECT_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/eager/local_call_opkernel_phy_instr_operand.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {

namespace vm {

class LocalCallOpKernelPhyInstrOperand;
class DTRInstrOperand;

class TensorBuffer {
 public:
  char* blob_dptr() { return blob_dptr_.get(); }
  void set_blob_dptr(std::unique_ptr<char, std::function<void(char*)>>&& blob_dptr) {
    blob_dptr_ = std::move(blob_dptr);
  }

  void reset() { blob_dptr_.reset(); }

 private:
  std::unique_ptr<char, std::function<void(char*)>> blob_dptr_;
};

class EagerBlobObject : public BlobObject {
 public:
  EagerBlobObject(const EagerBlobObject&) = delete;
  EagerBlobObject(EagerBlobObject&&) = delete;
  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  DataType data_type, const std::shared_ptr<TensorBuffer>& tensor_buffer)
      : EagerBlobObject(mem_case, shape, data_type, tensor_buffer, Optional<LocalDepObject*>()) {}

  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  DataType data_type, const std::shared_ptr<TensorBuffer>& tensor_buffer,
                  LocalDepObject* dep_object)
      : EagerBlobObject(mem_case, shape, data_type, tensor_buffer,
                        Optional<LocalDepObject*>(dep_object)) {}

  ~EagerBlobObject() override {
    non_pod_initer_.reset();
    tensor_buffer_.reset();
    header_buffer_.reset();
    blob_.reset();
  }

  std::vector<float> backup_data_;
  float hash_ = -1;

  BlobDesc* mut_blob_desc() override { return &blob_desc_; }
  std::size_t BlobBodyBytes() { return blob_body_bytes_; }

  const Blob& blob() const override { return *blob_; }
  Blob* mut_blob() override { return blob_.get(); }
  Maybe<void> TryInitBlob() override;
  Maybe<void> InitBlob();

  Maybe<void> TryAllocateBlobBodyMemory(DeviceCtx* device_ctx) override;
  Maybe<void> DeallocateBlobDataPtr() override {
    non_pod_initer_.reset();
    tensor_buffer_->reset();
    return Maybe<void>::Ok();
  }

  Maybe<LocalDepObject*> compute_local_dep_object() const {
    return JUST(compute_local_dep_object_);
  }

  std::shared_ptr<TensorBuffer>& tensor_buffer() { return tensor_buffer_; }
  char* object_dptr() { return tensor_buffer_->blob_dptr(); }

  bool is_shape_synced() const { return is_shape_synced_; }

  void set_is_shape_synced(bool val) { is_shape_synced_ = val; }

  const Optional<Symbol<Device>>& producer_op_device() const { return producer_op_device_; }
  Maybe<void> init_producer_op_device(Symbol<Device> producer_op_device) {
    CHECK_OR_RETURN(!producer_op_device_.has_value());
    producer_op_device_ = producer_op_device;
    return Maybe<void>::Ok();
  }

  const Optional<Symbol<Device>>& last_used_device() const { return last_used_device_; }
  void set_last_used_device(Symbol<Device> last_used_device) {
    last_used_device_ = last_used_device;
  }

  double blob_body_bytes_double() const { return static_cast<double>(blob_body_bytes_); }

 private:
  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  DataType data_type, const std::shared_ptr<TensorBuffer>& tensor_buffer,
                  const Optional<LocalDepObject*>& dep_object);

 protected:
  std::unique_ptr<Blob> blob_;
  std::unique_ptr<char[]> header_buffer_;
  std::shared_ptr<TensorBuffer> tensor_buffer_;
  std::size_t blob_body_bytes_;
  std::unique_ptr<MemoryAllocator> non_pod_initer_;
  std::atomic<bool> is_shape_synced_;
  Optional<LocalDepObject*> compute_local_dep_object_;
  Optional<Symbol<Device>> producer_op_device_;
  Optional<Symbol<Device>> last_used_device_;
};

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
                     const std::shared_ptr<TensorBuffer>& tensor_buffer, LocalDepObject* dep_object)
      : EagerBlobObject(mem_case, shape, data_type, tensor_buffer, dep_object),
        could_evict_(true),
        is_bp_required_(false),
        compute_time_(0),
        last_access_time_(0),
        pinned_(0),
        recompute_mode_(1),
        compute_op_(nullptr) {}
  ~DTREagerBlobObject() override;

  Maybe<void> InitBlobAttrs(std::shared_ptr<LocalCallOpKernelPhyInstrOperand>& operand);

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
  void update_user_ops(std::shared_ptr<LocalCallOpKernelPhyInstrOperand>& operand);
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

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_EAGER_BLOB_OBJECT_H_
