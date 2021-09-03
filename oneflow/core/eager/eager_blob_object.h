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
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/framework/vm_local_dep_object.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/eager/local_call_opkernel_phy_instr_operand.h"

namespace oneflow {

namespace vm {

class LocalCallOpKernelPhyInstrOperand;

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
      : EagerBlobObject(mem_case, shape, data_type, tensor_buffer, nullptr) {}
  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  DataType data_type, const std::shared_ptr<TensorBuffer>& tensor_buffer,
                  const std::shared_ptr<const ParallelDesc>& parallel_desc);
  ~EagerBlobObject() override {
    non_pod_initer_.reset();
    tensor_buffer_.reset();
    header_buffer_.reset();
    blob_.reset();
  }

  BlobDesc* mut_blob_desc() override { return &blob_desc_; }

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

  Maybe<VmLocalDepObject> compute_local_dep_object() const { return compute_local_dep_object_; }

  std::shared_ptr<TensorBuffer>& tensor_buffer() { return tensor_buffer_; }

  bool is_shape_synced() const { return is_shape_synced_; }

  void set_is_shape_synced(bool val) { is_shape_synced_ = val; }

 protected:
  std::unique_ptr<Blob> blob_;
  std::unique_ptr<char, std::function<void(char*)>> header_buffer_;
  std::shared_ptr<TensorBuffer> tensor_buffer_;
  std::size_t blob_body_bytes_;
  std::unique_ptr<MemoryAllocator> non_pod_initer_;
  std::atomic<bool> is_shape_synced_;
  Maybe<VmLocalDepObject> compute_local_dep_object_;
};

class DisjNode {
 public:
  DisjNode(double time) : compute_time_(time), parent_(nullptr) {}

  bool is_root() {
    return !bool(parent_);
  }

  void set_parent(std::shared_ptr<DisjNode>& parent) {
    parent_ = parent;
  }

 private:
  double compute_time_;
  std::shared_ptr<DisjNode> parent_;
};

class DTREagerBlobObject final : public EagerBlobObject {
 public:
  DTREagerBlobObject(const DTREagerBlobObject&) = delete;
  DTREagerBlobObject(DTREagerBlobObject&&) = delete;
  DTREagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  DataType data_type, const std::shared_ptr<TensorBuffer>& tensor_buffer)
      : DTREagerBlobObject(mem_case, shape, data_type, tensor_buffer, nullptr) {
        compute_time_ = 0;
        last_access_time_ = 0;
        pinned_ = 0;
        compute_op_ = nullptr;
        node_ = nullptr;
        user_ops_ = std::vector<std::shared_ptr<LocalCallOpKernelPhyInstrOperand>>();
      }
  DTREagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  DataType data_type, const std::shared_ptr<TensorBuffer>& tensor_buffer,
                  const std::shared_ptr<const ParallelDesc>& parallel_desc) : EagerBlobObject(mem_case, shape, data_type, tensor_buffer, parallel_desc) {
                    compute_time_ = 0;
                    last_access_time_ = 0;
                    pinned_ = 0;
                    compute_op_ = nullptr;
                    node_ = nullptr;
                    user_ops_ = std::vector<std::shared_ptr<LocalCallOpKernelPhyInstrOperand>>();
                  }
  ~DTREagerBlobObject() override {
    non_pod_initer_.reset();
    tensor_buffer_.reset();
    blob_.reset();
  }

  Maybe<void> InitBlobAttrs(std::shared_ptr<LocalCallOpKernelPhyInstrOperand>& operand);

  // Getters and Setters
  const std::size_t memory() const { return blob_body_bytes_; }
  const double compute_time() const { return compute_time_; }
  const double last_access_time() const { return last_access_time_; }
  std::shared_ptr<LocalCallOpKernelPhyInstrOperand> compute_op() const { return compute_op_; }
  void set_compute_time(double val) {
    if (val > 0) {
      compute_time_ = val;
    } else {
      compute_time_ = blob_body_bytes_;
    }
    // std::cout << "Compute time: " << compute_time_ << std::endl;
  }
  void set_last_access_time(double val) { last_access_time_ = val; }

  // DTR Strategy
  bool is_in_memory();
  bool is_pinned() { return (pinned_ > 0); }
  void pin() { pinned_++; std::cout << "pinned" << std::endl; }
  void unpin() { pinned_--; std::cout << "unpinned" << std::endl; }
  void update_access_time();
  void update_user_ops(std::shared_ptr<LocalCallOpKernelPhyInstrOperand>& operand);
  Maybe<void> evict() {
    evict_flag_ = true;
    DeallocateBlobDataPtr();
    return Maybe<void>::Ok();
    }
  Maybe<bool> execute() {
    evict_flag_ = false;
    return false;   // return false to enter find_best_and_evict()
  }
  Maybe<double> parent_cost();
  Maybe<double> child_cost();
  Maybe<double> neighbor_cost();

  // TODO: variable cost functions in terms of different heuristics
  Maybe<double> cost();

  std::shared_ptr<DisjNode> node_;

 private:
  bool evict_flag_ = false;
  double compute_time_;
  double last_access_time_;
  size_t pinned_;
  std::shared_ptr<LocalCallOpKernelPhyInstrOperand> compute_op_;
  std::vector<std::shared_ptr<LocalCallOpKernelPhyInstrOperand>> user_ops_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_EAGER_BLOB_OBJECT_H_
