#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INFER_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INFER_UTIL_H_

#include "oneflow/core/common/maybe.h"

namespace oneflow {

class Shape;
enum DataType;
class UserOpDef;
class UserOpConf;

namespace user_op {

using Shape4ArgNameAndIndexFn = std::function<Shape*(const std::string&, int32_t)>;
using Dtype4ArgNameAndIndexFn = std::function<DataType*(const std::string&, int32_t)>;
using ArgVec = std::vector<std::pair<std::string, int32_t>>;

class InferContext final {
 public:
  InferContext() = default;
  InferContext(Shape4ArgNameAndIndexFn, Dtype4ArgNameAndIndexFn, std::function<const ArgVec&()>,
               std::function<const ArgVec&()>);
  ~InferContext() = default;
  InferContext(const InferContext&) = delete;
  InferContext(InferContext&&) = delete;

  Shape* Shape4ArgNameAndIndex(const std::string&, int32_t) const;
  DataType* Dtype4ArgNameAndIndex(const std::string&, int32_t) const;
  const ArgVec& inputs() const { return inputs_(); }
  const ArgVec& outputs() const { return outputs_(); }

 private:
  Shape4ArgNameAndIndexFn shape_infer_fn_;
  Dtype4ArgNameAndIndexFn dtype_infer_fn_;
  std::function<const ArgVec&()> inputs_;
  std::function<const ArgVec&()> outputs_;
};

struct ShapeInferFnUtil {
  static Maybe<void> Unchanged(const InferContext&);
  static Maybe<void> InOutCorrespond(const InferContext&);
};

struct DtypeInferFnUtil {
  static Maybe<void> Unchanged(const InferContext&);
  static Maybe<void> InOutCorrespond(const InferContext&);
};

struct CheckAttrFnUtil {
  static Maybe<void> NoCheck(const UserOpDef&, const UserOpConf&);
};

}  // namespace user_op

}  // namespace oneflow

#endif
