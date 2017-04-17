#ifndef _CONTEXT_SOLVER_DESCRIPTOR_H_
#define _CONTEXT_SOLVER_DESCRIPTOR_H_
#include <glog/logging.h>
#include <cstdint>
#include <vector>
#include <memory>

/*
Parsed content from solver proto.
*/
namespace oneflow {
class SolverProto;
class SolverDescriptor {
 public:
  explicit SolverDescriptor(const oneflow::SolverProto& solver);
  ~SolverDescriptor() {}
  int32_t machine_id() const;
  int32_t max_iter() const;
  int32_t num_data_param_copy() const;
  int32_t num_model_param_copy() const;
  int32_t num_batch_per_sync() const;
  std::string train_net() const;

 private:
  std::string train_net_path_;
  int32_t max_iter_;
  int32_t machine_id_;
  int32_t num_data_param_copy_;
  int32_t num_model_param_copy_;
  int32_t num_batch_per_sync_;
  SolverDescriptor(const SolverDescriptor& other) = delete;
  SolverDescriptor& operator=(const SolverDescriptor& other) = delete;
};
}  // namespace oneflow
#endif  // _CONTEXT_SOLVER_DESCRIPTOR_H_
