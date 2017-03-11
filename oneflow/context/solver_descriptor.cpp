#include "context/solver_descriptor.h"
#include "caffe.pb.h"

namespace caffe {
SolverDescriptor::SolverDescriptor(const caffe::SolverProto& solver) {
  CHECK(solver.has_machine_id());
  machine_id_ = solver.machine_id();
  if (solver.has_max_iter()) {
    max_iter_ = solver.max_iter();
  }

  if (solver.has_num_data_param_copy()) {
    num_data_param_copy_ = solver.num_data_param_copy();
  } else {
    num_data_param_copy_ = 1;
  }

  if (solver.has_num_model_param_copy()) {
    num_model_param_copy_ = solver.num_model_param_copy();
  } else {
    num_model_param_copy_ = 1;
  }

  if (solver.has_num_batch_per_sync()) {
    num_batch_per_sync_ = solver.num_batch_per_sync();
  } else {
    num_batch_per_sync_ = 1;
  }

  if (solver.has_train_net()) {
    train_net_path_ = solver.train_net();
  }
}

int32_t SolverDescriptor::max_iter() const {
  return max_iter_;
}

int32_t SolverDescriptor::machine_id() const {
  return machine_id_;
}

int32_t SolverDescriptor::num_data_param_copy() const {
  return num_data_param_copy_;
}

int32_t SolverDescriptor::num_model_param_copy() const {
  return num_model_param_copy_;
}

int32_t SolverDescriptor::num_batch_per_sync() const {
  return num_batch_per_sync_;
}

std::string SolverDescriptor::train_net() const {
  return train_net_path_;
}
}  // namespace caffe
