#include "oneflow/core/eager/dtr_eager_blob_object.h"

#include "oneflow/core/eager/dtr_util.h"
#include "oneflow/core/framework/shut_down_util.h"
#include "oneflow/core/framework/tensor_pool.h"
#include "oneflow/core/eager/local_call_opkernel_phy_instr_operand.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/vm/thread_safe_allocator.h"
#include "oneflow/core/vm/dtr_cuda_allocator.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"

namespace oneflow {

namespace vm {

Maybe<void> DTREagerBlobObject::TryAllocateBlobBodyMemory(DeviceCtx* device_ctx) {
  vm::Allocator* allocator = device_ctx->mut_allocator();
  CHECK_NOTNULL_OR_RETURN(allocator);
  size_t required_body_bytes = AlignedByteSizeOfBlobBody();
  if (required_body_bytes == 0) {
    CHECK_ISNULL_OR_RETURN(tensor_storage_->blob_dptr());
    if (dtr::is_enabled_and_debug()) {
      LOG(INFO) << "ebo " << this << " has no body";
      LOG(INFO) << shape();
    }
    return Maybe<void>::Ok();
  }
  if (tensor_storage_->blob_dptr() != nullptr) {
    CHECK_EQ_OR_RETURN(tensor_storage_->blob_bytes(), required_body_bytes);
    if (dtr::debug_level() >= 2) {
      LOG(INFO) << "ebo " << this
                << " body already allocated, blob_body_bytes_: " << tensor_storage_->blob_bytes()
                << ", required_body_bytes: " << required_body_bytes;
    }
    return Maybe<void>::Ok();
  }
  {
    // reset tensor_buffer_;
    const auto& Free = [allocator, required_body_bytes](char* dptr) {
      if (IsShuttingDown()) { return; }
      allocator->Deallocate(dptr, required_body_bytes);
    };
    char* dptr = nullptr;
    allocator->Allocate(&dptr, required_body_bytes);
    if (ParseBooleanFromEnv("OF_DTR_ALLO", true)) {
      if (auto* b_allocator = dynamic_cast<vm::ThreadSafeAllocator*>(allocator)) {
        if (auto* dtr_allocator =
                dynamic_cast<vm::DtrCudaAllocator*>(b_allocator->backend_allocator())) {
          if (auto* dtr_ebo = dynamic_cast<vm::DTREagerBlobObject*>(this)) {
            dtr_allocator->Mark(dtr_ebo, dptr);
          } else {
            // do nothing
            if (dtr::is_enabled_and_debug()) {
              LOG(INFO) << "dtr_allocator has a non DTREagerBlobObject, " << typeid(*this).name();
            }
          }
        } else {
          OF_RUNTIME_ERROR() << "not dtr allocator, " << typeid(*b_allocator->backend_allocator()).name();
        }
      }
    }
    CHECK_NOTNULL_OR_RETURN(dptr);
    tensor_storage_->set_blob_dptr(std::unique_ptr<char, std::function<void(char*)>>(dptr, Free),
                                   required_body_bytes);
    InitNonPODTypeEagerBlobObjectIfNeed(tensor_storage_->non_pod_allocator(), this);
  }
  // rewrite it
  return Maybe<void>::Ok();
}

DTREagerBlobObject::DTREagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case,
                                       const std::shared_ptr<Shape>& shape, DataType data_type,
                                       const std::shared_ptr<TensorStorage>& tensor_storage,
                                       const intrusive::shared_ptr<LocalDepObject>& dep_object)
    : EagerBlobObject(mem_case, shape, data_type, tensor_storage, dep_object),
      could_evict_(true),
      is_bp_required_(false),
      compute_time_(0),
      last_access_time_(0),
      pinned_(0),
      recompute_mode_(1),
      compute_op_(nullptr) {
  node = std::make_shared<DisjNode>(0);
  auto pesudo_node = std::make_shared<DisjNode>(0);
  node->set_pesudo_node(pesudo_node);  // might induce some problems
}

DTREagerBlobObject::~DTREagerBlobObject() { clear_invalid_object(); }

void DTREagerBlobObject::pin() {
  pinned_++;
  if (dtr::debug_level() >= 2) {
    LOG(INFO) << "pinned " << this << ", " << (pinned_ - 1) << " to " << pinned_ << std::endl;
  }
}

void DTREagerBlobObject::unpin() {
  CHECK_GT(pinned_, 0) << this;
  pinned_--;
  if (dtr::debug_level() >= 2) {
    LOG(INFO) << "unpinned " << this << ", " << (pinned_ + 1) << " to " << pinned_ << std::endl;
  }
}

Maybe<void> DTREagerBlobObject::evict() {
  if (dtr::is_enabled_and_debug()) {
    static size_t c = 0;
    LOG(INFO) << "evict (No." << c++ << ")" << std::endl;
    LOG(INFO) << "this:" << this << std::endl;
  }
  CHECK_OR_RETURN(is_evictable());
  if (shape().elem_cnt() == 0) {
    if (dtr::is_enabled_and_debug()) {
      LOG(INFO) << "but elem_cnt is 0, shape is " << shape() << ", skip";
    }
    return Maybe<void>::Ok();
  }
  evict_flag_ = true;
  // NOTE: DeallocateBlobDataPtr() resets tensor_storage_, which is not we want
  tensor_storage_->Release();
  if (EnvBool<ONEFLOW_DTR_OPERATION_LOG>()) {
    LOG(INFO) << "****"
              << "EVICT-" << this << "-" << BlobBodyBytes() << std::endl;
  }
  if (blob_) { blob_->reset_dptr(nullptr); }
  CHECK_OR_RETURN(!is_in_memory());
  Global<dtr::TensorPool>::Get()->inc_num_eviction();
  return Maybe<void>::Ok();
}

void DTREagerBlobObject::clear_invalid_object() {
  if (IsShuttingDown()) { return; }
  CHECK_JUST(Global<dtr::TensorPool>::Get()->clear());
}

void DTREagerBlobObject::set_compute_op(LocalCallOpKernelPhyInstrOperand* operand) {
  update_access_time();
  compute_op_ = std::make_unique<DTRInstrOperand>(
      operand->shared_opkernel(), operand->inputs(), operand->outputs(),
      operand->consistent_tensor_infer_result(), operand->op_interp_ctx(),
      operand->dev_vm_dep_object_consume_mode());
  if (dtr::debug_level() >= 2) {
    LOG(INFO) << "set compute_op_ of " << this << " to " << compute_op_.get();
  }
  could_evict_ = (input_size() > 0) && could_evict_;
}

void DTREagerBlobObject::update_access_time() {
  last_access_time_ = Global<dtr::TensorPool>::Get()->duration();
  if (dtr::debug_level() >= 2) { LOG(INFO) << "update_access_time to " << last_access_time_; }
}

void DTREagerBlobObject::AppendUserOp(vm::LocalCallOpKernelPhyInstrOperand* operand) {
  user_ops_.emplace_back(std::make_unique<DTRInstrOperand>(
      operand->shared_opkernel(), operand->inputs(), operand->outputs(),
      operand->consistent_tensor_infer_result(), operand->op_interp_ctx(),
      operand->dev_vm_dep_object_consume_mode()));
}

bool DTREagerBlobObject::is_in_memory() const {
  // return !evict_flag_;
  return tensor_storage_->blob_dptr() != nullptr || shape().elem_cnt() == 0;
}

int DTREagerBlobObject::parent_depth() const {
  int max = -1;
  auto* ptr = dynamic_cast<DTRInstrOperand*>(compute_op_.get());
  CHECK_NOTNULL(ptr);
  for (const auto& input : ptr->inputs()) {
    if (auto object = input.lock()) {
      const auto dtr_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(object);
      CHECK_NOTNULL(dtr_blob_object);
      if (!dtr_blob_object->is_in_memory()) {
        max = std::max(max, dtr_blob_object->parent_depth());
      }
    }
  }
  return max + 1;
}

Maybe<double> DTREagerBlobObject::parent_cost(bool is_bp_required) const {
  double cost = 0;

  auto* ptr = dynamic_cast<DTRInstrOperand*>(compute_op_.get());
  CHECK_NOTNULL_OR_RETURN(ptr);
  for (const auto& input : ptr->inputs()) {
    if (!input.expired()) {
      auto object = input.lock();
      const auto dtr_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(object);
      CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
      bool add_flag = (!dtr_blob_object->is_in_memory());
      if (is_bp_required) { add_flag = add_flag && dtr_blob_object->is_bp_required(); }
      if (add_flag) {
        auto com_time = dtr_blob_object->compute_time();
        auto p_cost = JUST(dtr_blob_object->parent_cost(is_bp_required));
        cost = cost + com_time + p_cost;
      }
    }

    // CHECK_OR_RETURN(static_cast<bool>(input.get()));
    // auto object = input.get();
    // const auto* dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(input.get());
    // // CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
    // if (dtr_blob_object == nullptr) {
    //   continue;
    // } else {
    //   if (!dtr_blob_object->is_in_memory()) {
    //     auto com_time = dtr_blob_object->compute_time();
    //     auto p_cost = JUST(dtr_blob_object->parent_cost());
    //     cost = cost + com_time + p_cost;
    //   }
    // }
  }

  return cost;
}

int DTREagerBlobObject::child_depth() const {
  int max = -1;
  for (int i = 0; i < user_ops_.size(); ++i) {
    const auto* ptr = dynamic_cast<DTRInstrOperand*>(CHECK_JUST(user_op(i)));
    CHECK_NOTNULL(ptr);
    for (const auto& output : ptr->outputs()) {
      if (auto object = output.lock()) {
        const auto dtr_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(object);
        CHECK_NOTNULL(dtr_blob_object);
        if (!dtr_blob_object->is_in_memory()) {
          max = std::max(max, dtr_blob_object->child_depth());
        }
      }
    }
  }
  return max + 1;
}

Maybe<double> DTREagerBlobObject::child_cost(bool is_bp_required) const {
  double cost = 0;

  for (int i = 0; i < user_ops_.size(); ++i) {
    const auto* ptr = dynamic_cast<DTRInstrOperand*>(JUST(user_op(i)));
    CHECK_NOTNULL_OR_RETURN(ptr);
    for (const auto& output : ptr->outputs()) {
      if (!output.expired()) {
        auto object = output.lock();
        const auto dtr_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(object);
        CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
        bool add_flag = (!dtr_blob_object->is_in_memory());
        if (is_bp_required) { add_flag = add_flag && dtr_blob_object->is_bp_required(); }
        if (add_flag) {
          auto com_time = dtr_blob_object->compute_time();
          auto c_cost = JUST(dtr_blob_object->child_cost(is_bp_required));
          cost = cost + com_time + c_cost;
        }
      }

      // CHECK_OR_RETURN(static_cast<bool>(output.get()));
      // auto object = output.get();
      // CHECK_NOTNULL_OR_RETURN(object);
      // const auto* dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(output.get());
      // // CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
      // if (dtr_blob_object == nullptr) {
      //   continue;
      // } else {
      //   if (!dtr_blob_object->is_in_memory()) {
      //     auto com_time = dtr_blob_object->compute_time();
      //     auto c_cost = JUST(dtr_blob_object->child_cost());
      //     cost = cost + com_time + c_cost;
      //   }
      // }
    }
  }

  return cost;
}

Maybe<double> DTREagerBlobObject::neighbor_cost() const {
  const auto p_cost = JUST(parent_cost());
  const auto c_cost = JUST(child_cost());
  return p_cost + c_cost + compute_time_;
}

Maybe<double> DTREagerBlobObject::approx_neighbor_cost() const {
  double cost = 0;
  const auto& inputs = compute_op_->inputs();
  for (int i = 0; i < inputs.size(); ++i) {
    if (auto tmp = inputs[i].lock()) {
      auto dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(tmp.get());
      CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
      if (!dtr_blob_object->is_in_memory()) {
        double p_cost =
            Global<dtr::TensorPool>::Get()->find_father(dtr_blob_object->node)->compute_time();
        if (p_cost < dtr_blob_object->compute_time()) { p_cost = dtr_blob_object->compute_time(); }
        cost += p_cost;
      }
    }
  }

  const auto& outputs = compute_op_->outputs();
  for (int i = 0; i < outputs.size(); ++i) {
    if (auto tmp = outputs[i].lock()) {
      auto dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(tmp.get());
      CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
      if (!dtr_blob_object->is_in_memory()) {
        double c_cost =
            Global<dtr::TensorPool>::Get()->find_father(dtr_blob_object->node)->compute_time();
        if (c_cost < dtr_blob_object->compute_time()) { c_cost = dtr_blob_object->compute_time(); }
        cost += c_cost;
      }
    }
  }

  return cost + compute_time_;
}

Maybe<double> DTREagerBlobObject::cost(const std::string& heuristic) const {
  const double time_since_last_access =
      heuristic == "size" ? 1 : Global<dtr::TensorPool>::Get()->duration() - last_access_time_;

  if (dtr::debug_level() >= 2) {
    std::cout << std::dec << "ap compute " << JUST(approx_neighbor_cost()) << ", blob_body_bytes_ "
              << tensor_storage_->blob_bytes() << ", time_since_last_access "
              << time_since_last_access << std::endl;
    // const auto pd = parent_depth();
    // const auto cd = child_depth();
    // std::cout << "parent depth: " << pd << ", child depth: " << cd << ", total depth: " << pd +
    // cd
    // << std::endl;
  }
  if (heuristic == "random") {
    return static_cast<double>(rand()) / RAND_MAX;
  } else if (heuristic == "size") {
    return 1 / blob_body_bytes_double();
  } else if (heuristic == "full") {
    return JUST(neighbor_cost()) / blob_body_bytes_double() / time_since_last_access;
  } else if (heuristic == "eq") {
    return JUST(approx_neighbor_cost()) / blob_body_bytes_double() / time_since_last_access;
  } else if (heuristic == "bp_aware") {
    return reverse_cost();
  } else if (heuristic == "depth") {
    return parent_depth() + child_depth();
  } else if (heuristic == "local") {
    return compute_time_ / blob_body_bytes_double() / time_since_last_access;
  } else if (heuristic == "lru") {
    return 1 / time_since_last_access;
  } else if (heuristic == "compute_time_and_size") {
    return JUST(neighbor_cost()) / blob_body_bytes_double();
  } else if (heuristic == "compute_time") {
    return JUST(neighbor_cost());
  } else if (heuristic == "eq_compute_time_and_last_access") {
    return JUST(approx_neighbor_cost()) / time_since_last_access;
  } else if (heuristic == "local_compute_time_and_last_access") {
    return compute_time_ / time_since_last_access;
  } else if (heuristic == "local_compute_time") {
    return compute_time_;
  } else {
    return Error::InvalidValueError("");
  }
}

Maybe<double> DTREagerBlobObject::cost() const {
  const auto& heuristic = Global<DTRConfig>::Get()->heuristic;
  const double cost = JUST(this->cost(heuristic));
  return cost;
}

void DTREagerBlobObject::set_compute_time(double val) {
  if (val > 0) {
    // TODO: add a minimal cost for bytes == 0?
    compute_time_ = val;
  } else {
    compute_time_ = tensor_storage_->blob_bytes();
  }
  if (ParseBooleanFromEnv("OF_DTR_HIGH_ADD_N", true)) {
    if (compute_op_type_name() == "add_n") { compute_time_ *= 5; }
  }
  if (ParseBooleanFromEnv("OF_DTR_HIGH_CONV", true)) {
    if (compute_op_type_name() == "conv2d") { compute_time_ *= 5; }
    if (compute_op_type_name() == "conv_filter_grad") { compute_time_ *= 5; }
    if (compute_op_type_name() == "conv_data_grad") { compute_time_ *= 5; }
  }
  if (dtr::debug_level() >= 2) {
    LOG(INFO) << "Compute time of " << this << ": " << compute_time_ << ", compute op "
              << compute_op_type_name() << std::endl;
  }
}

Maybe<double> DTREagerBlobObject::reverse_cost() const {
  double cost = JUST(rev_fwd_cost());
  double bwd_cost = JUST(rev_bwd_cost());

  if (bwd_cost < cost) {
    set_recompute_mode(-2);
    cost = bwd_cost;
  }
  return cost;
}

Maybe<double> DTREagerBlobObject::rev_fwd_cost() const {
  // base_cost
  double time_since_last_access = Global<dtr::TensorPool>::Get()->duration() - last_access_time_;
  double base_cost = compute_time_ / tensor_storage_->blob_bytes() / time_since_last_access;

  // parent_cost for compute_op_
  // parent_cost: sum of cost for all parent nodes that are not in memory
  double cost_parent = compute_time_ + JUST(parent_cost(true));

  // sum of the cost for all potentials tensors that will be recomputed and stored in memory after
  // the recomputation of the current tensor
  double cost_fwd_potential = compute_time_;
  size_t available_bytes =
      dtr::memory_threshold() - Global<dtr::TensorPool>::Get()->get_total_memory();
  size_t tmp_bytes = 0;
  auto* ptr = dynamic_cast<DTRInstrOperand*>(compute_op_.get());
  CHECK_NOTNULL_OR_RETURN(ptr);
  for (const auto& input : ptr->inputs()) {
    if (!input.expired()) {
      auto object = input.lock();
      const auto dtr_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(object);
      CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
      if (tmp_bytes + dtr_blob_object->compute_time() < available_bytes
          && !dtr_blob_object->is_in_memory() && dtr_blob_object->is_bp_required()) {
        tmp_bytes += dtr_blob_object->compute_time();
      } else {
        break;
      }
    }
  }
  cost_fwd_potential += tmp_bytes;

  return base_cost * cost_parent / cost_fwd_potential;
}

Maybe<double> DTREagerBlobObject::rev_bwd_cost() const {
  double time_since_last_access = Global<dtr::TensorPool>::Get()->duration() - last_access_time_;
  double base_cost = compute_time_ / tensor_storage_->blob_bytes() / time_since_last_access;

  // parent_cost for user_op
  // parent_cost: sum of cost for all parent nodes that are not in memory
  double cost_parent = -1;
  for (int i = 0; i < user_ops_.size(); ++i) {
    double tmp_cost = 0;
    const auto* ptr = dynamic_cast<DTRInstrOperand*>(JUST(user_op(i)));
    CHECK_NOTNULL_OR_RETURN(ptr);
    for (const auto& output : ptr->outputs()) {
      if (!output.expired()) {
        auto object = output.lock();
        const auto dtr_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(object);
        CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
        if (!dtr_blob_object->is_in_memory() && dtr_blob_object->is_bp_required()) {
          auto com_time = dtr_blob_object->compute_time();
          auto c_cost = JUST(dtr_blob_object->child_cost());
          tmp_cost = tmp_cost + com_time + c_cost;
        }
      }
    }
    if (cost_parent < 0 || tmp_cost < cost_parent) { cost_parent = tmp_cost; }
  }
  cost_parent += compute_time_;

  double cost_fwd_potential = compute_time_;

  return base_cost * cost_parent / cost_fwd_potential;
}

size_t DTREagerBlobObject::input_size() const { return compute_op_->inputs().size(); }

const std::string& DTREagerBlobObject::compute_op_type_name() const {
  static std::string no_compute_op = "no compute op";
  if (!compute_op_) {
    if (dtr::is_enabled_and_debug()) { LOG(INFO) << "no compute op for " << this; }
    return no_compute_op;
  }
  return compute_op_->shared_opkernel()->op_type_name();
}

bool DTREagerBlobObject::is_evictable() const {
  if (!compute_op_) { return false; }
  if (compute_op_->inputs().empty()) { return false; }
  // FIXME: set_tensor_inputs should also include other outputs of the compute_op
  if (compute_op_->shared_opkernel()->user_op_conf_->op_type_name() == "nll") { return false; }
  if (compute_op_->shared_opkernel()->user_op_conf_->op_type_name() == "copy") { return false; }
  // if (compute_op_->shared_opkernel()->user_op_conf_->op_type_name() == "conv_filter_grad") {
  // return false; } if (compute_op_->shared_opkernel()->user_op_conf_->op_type_name() ==
  // "matmul") { return false; }
  return could_evict_;
}

void DTREagerBlobObject::reset_pesudo_node() { node->reset_pesudo_node(); }

void DisjNode::reset_pesudo_node() {
  auto pesudo_node_ = std::make_shared<DisjNode>(compute_time_);
  if (parent_ != nullptr) {
    if (dtr::debug_level() >= 3) {
      LOG(INFO) << "Parent of node is not nullptr while reseting the pesudo node!!!";
    }
    auto&& p = parent_->pesudo_node();
    pesudo_node_->set_parent(p);
  }
}

}  // namespace vm
}  // namespace oneflow
