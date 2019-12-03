#ifndef ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/blob_def.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

class Shape;
enum DataType;

namespace user_op {

class UserOpDefWrapper;

using Shape4ArgNameAndIndexFn = std::function<Shape*(const std::string&, int32_t)>;
using Dtype4ArgNameAndIndexFn = std::function<DataType*(const std::string&, int32_t)>;
using ArgVec = std::vector<std::pair<std::string, int32_t>>;
using Arg2BlobDef = HashMap<std::pair<std::string, int32_t>, BlobDef>;

class InferContext final {
 public:
  InferContext(UserOpConfWrapper&&, Arg2BlobDef&&);
  ~InferContext() = default;
  InferContext(const InferContext&) = delete;
  InferContext(InferContext&&) = delete;

  Shape* Shape4ArgNameAndIndex(const std::string&, int32_t);
  DataType* Dtype4ArgNameAndIndex(const std::string&, int32_t);
  const ArgVec& inputs() const;
  const ArgVec& outputs() const;

  template<typename T>
  T GetAttr(const std::string& attr_name) const {
    return conf_.attr<T>(attr_name);
  }

 private:
  UserOpConfWrapper conf_;
  ArgVec inputs_;
  ArgVec outputs_;
  Arg2BlobDef arg2blob_def_;
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
  static size_t ZeroTmpSize(const InferContext&);
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_
