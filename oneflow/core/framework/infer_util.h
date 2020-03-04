#ifndef ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/tensor_desc.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

class Shape;
enum DataType;

namespace user_op {

class UserOpDefWrapper;

class InferContext {
 public:
  virtual ~InferContext() = default;

  virtual Shape* Shape4ArgNameAndIndex(const std::string&, int32_t) = 0;
  virtual DataType* Dtype4ArgNameAndIndex(const std::string&, int32_t) = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;

  template<typename T>
  T GetAttr(const std::string& attr_name) const {
    return conf_.attr<T>(attr_name);
  }

 protected:
  InferContext(UserOpConfWrapper&& conf) : conf_(std::move(conf)) {}
  InferContext(const InferContext&) = delete;
  InferContext(InferContext&&) = delete;

 private:
  UserOpConfWrapper conf_;
};

struct ShapeInferFnUtil {
  static Maybe<void> Unchanged(InferContext*);
  static Maybe<void> InOutCorrespond(InferContext*);
};

struct DtypeInferFnUtil {
  static Maybe<void> Unchanged(InferContext*);
  static Maybe<void> InOutCorrespond(InferContext*);
};

struct CheckAttrFnUtil {
  static Maybe<void> NoCheck(const UserOpDefWrapper&, const UserOpConfWrapper&);
};

struct TmpSizeInferFnUtil {
  static size_t ZeroTmpSize(InferContext*);
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_
