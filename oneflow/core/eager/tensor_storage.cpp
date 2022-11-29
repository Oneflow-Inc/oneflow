#include "oneflow/core/eager/tensor_storage.h"
#include "oneflow/core/vm/op_call_instruction_policy.h"
#include "oneflow/core/vm/dtr_env.h"

namespace oneflow {
namespace vm {
namespace {
int64_t unique_id_2() {
  static size_t id = 0;
  return id++;
}

static double GetEstimatedComputeTime(const OpCallInstructionPolicy& operand) {
  const auto& inputs = operand.inputs();
  const auto& outputs = operand.outputs();
  size_t estimated_compute_time = 0;
  for (const auto& input : inputs) {
    estimated_compute_time += input->tensor_storage()->blob_bytes();
  }
  for (const auto& output : outputs) {
    estimated_compute_time += output->tensor_storage()->blob_bytes();
  }
  return estimated_compute_time;
}

}  // namespace

TensorStorage::TensorStorage()
    : id_(unique_id_2()),
      num_pinned_(0),
      blob_bytes_(0),
      last_access_time_(0),
      compute_time_(0),
      non_pod_allocator_(std::make_unique<MemoryAllocator>()),
      producer_stream_(NullOpt),
      last_used_stream_(NullOpt) {
  VLOG(1) << "create storage " << id_;
}

TensorStorage::~TensorStorage() {
  for (const auto& hook : storage_delete_hooks_) { hook(); }
  if (compute_op_) { Singleton<dtr::Env>::Get()->remove_compute_op(compute_op_.get()); }
  VLOG(1) << "delete storage " << id_;
}

void TensorStorage::Evict(bool eager_eviction) {
  Singleton<dtr::Env>::Get()->add_eviction_num(eager_eviction);
  return Release();
}

OpCallInstructionPolicy TensorStorage::compute_op() const {
  CHECK_NOTNULL(compute_op_);
  return OpCallInstructionPolicy(*compute_op_);
}

void TensorStorage::clear_compute_op() {
  if (compute_op_ == nullptr) { return; }
  Singleton<dtr::Env>::Get()->remove_compute_op(compute_op_.get());
  compute_op_ = nullptr;
  compute_time_ = -1;
}

void TensorStorage::set_compute_op(const OpCallInstructionPolicy& compute_op) {
  if (compute_op_) {
    CHECK(false);
    // Singleton<dtr::Env>::Get()->remove_compute_op(compute_op_.get());
  }
  // copy a new OpCallInstructionPolicy
  compute_op_ = std::make_shared<DtrOpCallInstructionPolicy>(compute_op);
  Singleton<dtr::Env>::Get()->ops.push_back(compute_op_.get());
  compute_time_ = GetEstimatedComputeTime(compute_op);
}

std::string TensorStorage::compute_op_type_name() const {
  if (compute_op_) {
    return compute_op_->opkernel().op_type_name();
  } else {
    return "None";
  }
}

void TensorStorage::Access() { last_access_time_ = Singleton<dtr::Env>::Get()->time_now(); }

Maybe<double> TensorStorage::cost(size_t override_size) const {
  const double time_since_last_access = Singleton<dtr::Env>::Get()->time_now() - last_access_time_;
  size_t size = override_size == 0 ? blob_bytes_ : override_size;
  return compute_time_ / time_since_last_access;
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
