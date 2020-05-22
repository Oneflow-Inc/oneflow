#ifndef ONEFLOW_CUSTOMIZED_DATA_EMPTY_TENSOR_MANAGER_H_
#define ONEFLOW_CUSTOMIZED_DATA_EMPTY_TENSOR_MANAGER_H_

#include "oneflow/core/common/util.h"

namespace oneflow {
namespace data {

template<typename LoadTarget>
void PrepareEmptyTensor(LoadTarget& tensor, int64_t tensor_init_bytes);

template<typename LoadTarget>
class EmptyTensorManager final {
 public:
  using LoadTargetUniquePtr = std::unique_ptr<LoadTarget>;
  using LoadTargetSharedPtr = std::shared_ptr<LoadTarget>;

  EmptyTensorManager(int64_t total_empty_size, int64_t tensor_init_bytes);
  ~EmptyTensorManager() = default;

  void Recycle(LoadTarget* tensor_ptr);
  LoadTargetSharedPtr Get();

 private:
  const int64_t tensor_init_bytes_;
  int64_t total_tensor_count_;
  std::mutex empty_tensors_mutex_;
  std::vector<LoadTargetUniquePtr> empty_tensors_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_EMPTY_TENSOR_MANAGER_H_
