#ifndef ONEFLOW_CORE_FRAMEWORK_BATCH_AXIS_CONTEXT_H_
#define ONEFLOW_CORE_FRAMEWORK_BATCH_AXIS_CONTEXT_H_

#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

namespace user_op {

class TensorDesc;

class BatchAxisContext {
 public:
  virtual ~BatchAxisContext() = default;

  virtual const TensorDesc& LogicalTensorDesc4InputArgNameAndIndex(
      const std::string& input_arg_name, int32_t index) const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;

  virtual OptInt64* BatchAxis4ArgNameAndIndex(const std::string& arg_name, int32_t index) = 0;

  template<typename T>
  T Attr(const std::string& attr_name) const {
    return user_op_conf_.attr<T>(attr_name);
  }

 protected:
  BatchAxisContext(UserOpConfWrapper&& conf) : user_op_conf_(conf) {}
  BatchAxisContext(const BatchAxisContext&) = delete;

 private:
  UserOpConfWrapper user_op_conf_;
};

struct BatchAxisInferFnUtil {
  static Maybe<void> DefaultAsFirstHasValueInput(BatchAxisContext*);
  static Maybe<void> NaiveInferBatchAxis(BatchAxisContext*);
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_BATCH_AXIS_CONTEXT_H_
