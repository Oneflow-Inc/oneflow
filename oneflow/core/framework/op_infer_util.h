#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INFER_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INFER_UTIL_H_

#include "oneflow/core/common/maybe.h"

namespace oneflow {

class Shape;
enum DataType;
class UserOpConf;

namespace user_op {

class UserOpDefWrapper;
class UserOpConfWrapper;

using Shape4ArgNameAndIndexFn = std::function<Shape*(const std::string&, int32_t)>;
using Dtype4ArgNameAndIndexFn = std::function<DataType*(const std::string&, int32_t)>;
using ArgVec = std::vector<std::pair<std::string, int32_t>>;

class InferContext final {
 public:
  InferContext() = default;
  InferContext(Shape4ArgNameAndIndexFn, Dtype4ArgNameAndIndexFn, const UserOpConf*);
  ~InferContext() = default;
  InferContext(const InferContext&) = delete;
  InferContext(InferContext&&) = delete;

  Shape* Shape4ArgNameAndIndex(const std::string&, int32_t) const;
  DataType* Dtype4ArgNameAndIndex(const std::string&, int32_t) const;
  const ArgVec& inputs() const;
  const ArgVec& outputs() const;

  template<typename T>
  T GetAttr(const std::string&) const;

 private:
  const UserOpConf* conf_;
  ArgVec inputs_;
  ArgVec outputs_;
  Shape4ArgNameAndIndexFn shape_infer_fn_;
  Dtype4ArgNameAndIndexFn dtype_infer_fn_;
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
  static Maybe<void> NoCheck(const UserOpDefWrapper&, const UserOpConf&);
};

}  // namespace user_op

}  // namespace oneflow

#endif
