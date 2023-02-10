#include "oneflow/core/eager/tensor_storage.h"
#include "oneflow/core/common/env_var/dtr.h"
#include "oneflow/core/vm/op_call_instruction_policy.h"
#include "oneflow/core/vm/dtr_disjoint_set.h"
#include "oneflow/core/vm/dtr_env.h"
#include "oneflow/core/vm/virtual_machine.h"

namespace oneflow {
namespace vm {
namespace {
int64_t unique_id_2() {
  static size_t id = 0;
  return id++;
}

}  // namespace

TensorStorage::TensorStorage(bool is_allocated_in_vm)
    : blob_bytes_(0),
      non_pod_allocator_(std::make_unique<MemoryAllocator>()),
      producer_stream_(NullOpt),
      last_used_stream_(NullOpt),
      is_allocated_in_vm_(is_allocated_in_vm) {}

RematableTensorStorage::RematableTensorStorage()
    : TensorStorage(true),
      node(std::make_shared<dtr::DisjNode>(0)),
      id_(unique_id_2()),
      num_pinned_(0),
      last_access_time_(0),
      compute_time_(0) {
  VLOG(1) << "create rematable storage " << id_;
}

RematableTensorStorage::~RematableTensorStorage() {
  if (compute_op_) { Singleton<dtr::Env>::Get()->remove_compute_op(compute_op_.get()); }
  VLOG(1) << "delete storage " << id_;
}

Symbol<Device> TensorStorage::device() const { return device_; }

void RematableTensorStorage::LogEviction(bool eager_eviction) const {
  Singleton<dtr::Env>::Get()->add_eviction_num(eager_eviction);
  VLOG(1) << "evict storage " << id_ << ", compute op type: " << compute_op_type_name()
          << ", eager_eviction: " << eager_eviction;
}

void RematableTensorStorage::Remat() {
  if (is_in_memory()) { return; }
  auto stream = CHECK_JUST(GetDefaultStreamByDevice(device_));
  auto* vm_stream = CHECK_JUST(Singleton<VirtualMachine>::Get()->GetVmStream(stream));
  auto op = compute_op();
  CHECK_JUST(Recompute(&op, vm_stream));
}

void RematableTensorStorage::Evict(bool eager_eviction) {
  CHECK(!is_eviction_disabled());
  LogEviction(eager_eviction);
  return _Release();
}

void TensorStorage::_Release() {
  for (const auto& hook : storage_delete_hooks_) { hook(); }
  non_pod_allocator_.reset();
  blob_dptr_.reset();
}

void TensorStorage::Release() { return _Release(); }

void RematableTensorStorage::Release() {
  CHECK(device_->with_remat());
  if (is_eviction_disabled()) { return; }
  return Evict(true);
}

Maybe<void> TensorStorage::init_producer_stream(Symbol<::oneflow::Stream> producer_stream) {
  CHECK_OR_RETURN(!producer_stream_.has_value());
  producer_stream_ = producer_stream;
  device_ = producer_stream->device();
  VLOG(1) << "device remat: " << device_->with_remat() << ", repr: " << device_->ToRepr();
  return Maybe<void>::Ok();
}

std::vector<std::string> random_ops{"uniform", "uniform_int", "normal", "randperm"};

bool RematableTensorStorage::is_evictable() const {
  return compute_op_ != nullptr
         && std::find(random_ops.begin(), random_ops.end(), compute_op_type_name())
                == random_ops.end()
         && !eviction_disabled_;
}

OpCallInstructionPolicy RematableTensorStorage::compute_op() const {
  CHECK_NOTNULL(compute_op_);
  return OpCallInstructionPolicy(*compute_op_);
}

std::shared_ptr<DtrOpCallInstructionPolicy> RematableTensorStorage::dtr_compute_op() const {
  return compute_op_;
}

void RematableTensorStorage::Pin() {
  ++num_pinned_;
  VLOG(3) << "pin storage " << id_ << ", num_pinned: " << num_pinned_;
}

void RematableTensorStorage::Unpin() {
  CHECK_GT(num_pinned_, 0);
  --num_pinned_;
  VLOG(3) << "unpin storage " << id_ << ", num_pinned: " << num_pinned_;
}

void RematableTensorStorage::clear_compute_op() {
  if (compute_op_ == nullptr) { return; }
  VLOG(1) << "clear_compute_op: " << id_;
  Singleton<dtr::Env>::Get()->remove_compute_op(compute_op_.get());
  compute_op_ = nullptr;
  compute_time_ = -1;
}

void RematableTensorStorage::set_compute_op(
    const std::shared_ptr<DtrOpCallInstructionPolicy>& compute_op, double compute_time) {
  if (compute_op_) {
    CHECK(false);
    // Singleton<dtr::Env>::Get()->remove_compute_op(compute_op_.get());
  }
  compute_op_ = compute_op;
  VLOG(1) << "set_compute_op: " << id_ << ", compute op: " << compute_op.get();
  Singleton<dtr::Env>::Get()->ops.push_back(CHECK_NOTNULL(compute_op_.get()));
  compute_time_ = compute_time;
}

std::string RematableTensorStorage::compute_op_type_name() const {
  if (is_eviction_disabled()) { return "eviction_disabled"; }
  if (compute_op_) { return compute_op_->opkernel().op_type_name(); }
  return "None";
}

void RematableTensorStorage::Access() {
  last_access_time_ = Singleton<dtr::Env>::Get()->time_now();
}

Maybe<double> RematableTensorStorage::cost(size_t override_size) const {
  const double time_since_last_access = Singleton<dtr::Env>::Get()->time_now() - last_access_time_;
  size_t size = 1;
  if (EnvBool<ONEFLOW_DTR_HEURISTIC_DTE>() || EnvBool<ONEFLOW_DTR_HEURISTIC_DTR>()) {
    size = override_size == 0 ? blob_bytes_ : override_size;
  }
  return (EnvBool<ONEFLOW_DTR_NEIGHBOR>() ? approx_neighbor_cost() : compute_time_)
         / time_since_last_access / static_cast<double>(size);
}

double RematableTensorStorage::approx_neighbor_cost() const {
  double cost = 0;
  auto compute_op = this->compute_op();
  const auto& inputs = compute_op.inputs();
  for (int i = 0; i < inputs.size(); ++i) {
    const auto& tmp = inputs[i];
    if (auto storage = std::dynamic_pointer_cast<RematableTensorStorage>(tmp->tensor_storage());
        !storage->is_in_memory()) {
      double p_cost = dtr::DisjointSet::find_father(storage->node)->compute_time();
      if (p_cost < storage->compute_time()) { p_cost = storage->compute_time(); }
      cost += p_cost;
    }
  }

  const auto& outputs = compute_op.outputs();
  for (int i = 0; i < outputs.size(); ++i) {
    const auto& tmp = outputs[i];
    if (auto storage = std::dynamic_pointer_cast<RematableTensorStorage>(tmp->tensor_storage());
        !storage->is_in_memory()) {
      double c_cost = dtr::DisjointSet::find_father(storage->node)->compute_time();
      if (c_cost < storage->compute_time()) { c_cost = storage->compute_time(); }
      cost += c_cost;
    }
  }

  return cost + compute_time_;
}

#if false
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
        LOG(INFO) << "parent " << dtr_blob_object.get()
                  << " op type: " << dtr_blob_object->compute_op_type_name();
        auto com_time = dtr_blob_object->compute_time();
        auto p_cost = JUST(dtr_blob_object->parent_cost(is_bp_required));
        cost = cost + com_time + p_cost;
        LOG(INFO) << "parent " << dtr_blob_object.get()
                  << " op type: " << dtr_blob_object->compute_op_type_name() << " end";
      }
    }
  }

  return cost;
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
    }
  }

  return cost;
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

Maybe<double> DTREagerBlobObject::cost(const std::string& heuristic, size_t override_size) const {
  const double time_since_last_access =
      heuristic == "size" ? 1 : Global<dtr::TensorPool>::Get()->duration() - last_access_time_;

  const double size_d = [&]() {
    if (override_size == 0) {
      return blob_body_bytes_double();
    } else {
      return static_cast<double>(override_size);
    }
  }();

  if (dtr::debug_level() >= 2) {
    LOG(INFO) << std::dec << "ap compute " << JUST(approx_neighbor_cost()) << ", blob_body_bytes_ "
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
    return 1 / size_d;
  } else if (heuristic == "full") {
    return JUST(neighbor_cost()) / size_d / time_since_last_access;
  } else if (heuristic == "eq") {
    return JUST(approx_neighbor_cost()) / size_d / time_since_last_access;
  } else if (heuristic == "bp_aware") {
    return reverse_cost();
  } else if (heuristic == "depth") {
    return parent_depth() + child_depth();
  } else if (heuristic == "local") {
    return compute_time_ / size_d / time_since_last_access;
  } else if (heuristic == "lru") {
    return 1 / time_since_last_access;
  } else if (heuristic == "compute_time_and_size") {
    return JUST(neighbor_cost()) / size_d;
  } else if (heuristic == "compute_time") {
    return JUST(neighbor_cost());
  } else if (heuristic == "full_compute_time_and_last_access") {
    return JUST(neighbor_cost()) / time_since_last_access;
  } else if (heuristic == "eq_compute_time_and_last_access") {
    return JUST(approx_neighbor_cost()) / time_since_last_access;
  } else if (heuristic == "eq_compute_time") {
    return JUST(approx_neighbor_cost());
  } else if (heuristic == "local_compute_time_and_last_access") {
    return compute_time_ / time_since_last_access;
  } else if (heuristic == "local_compute_time") {
    return compute_time_;
  } else if (heuristic == "eq_mul_beta") {
    return (JUST(approx_neighbor_cost()) * std::pow(0.5, recompute_times()))
           / (size_d * time_since_last_access);
  } else if (heuristic == "eq_div_beta") {
    return JUST(approx_neighbor_cost())
           / (size_d * time_since_last_access * std::pow(0.5, recompute_times()));
  } else {
    return Error::InvalidValueError("");
  }
}
#endif

}  // namespace vm
}  // namespace oneflow
