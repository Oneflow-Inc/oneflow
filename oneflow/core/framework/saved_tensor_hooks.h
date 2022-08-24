#ifndef ONEFLOW_CORE_FRAMEWORK_SAVED_TENSOR_HOOKS_H_
#define ONEFLOW_CORE_FRAMEWORK_SAVED_TENSOR_HOOKS_H_

#include "oneflow/core/framework/tensor.h"

namespace oneflow {
namespace one {
class SavedTensorHook {
 public:
  virtual ~SavedTensorHook() = default;
  virtual void pack(const std::shared_ptr<Tensor>& tensor) = 0;
  virtual std::shared_ptr<Tensor> unpack() = 0;
};

class SavedTensorHookCreator {
 public:
  virtual ~SavedTensorHookCreator() = default;
  virtual std::unique_ptr<SavedTensorHook> new_saved_tensor_hook() const = 0;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SAVED_TENSOR_HOOKS_H_
