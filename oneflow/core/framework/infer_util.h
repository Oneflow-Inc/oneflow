#ifndef ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/blob_def.h"

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
using Arg2BlobDef = HashMap<std::pair<std::string, int32_t>, BlobDef>;

class InferContext final {
 public:
  InferContext(const UserOpConf*, Arg2BlobDef&&);
  ~InferContext() = default;
  InferContext(const InferContext&) = delete;
  InferContext(InferContext&&) = delete;

  Shape* Shape4ArgNameAndIndex(const std::string&, int32_t);
  DataType* Dtype4ArgNameAndIndex(const std::string&, int32_t);
  const ArgVec& inputs() const;
  const ArgVec& outputs() const;

  template<typename T>
  T GetAttr(const std::string&) const;

 private:
  const UserOpConf* conf_;
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
  static Maybe<void> NoCheck(const UserOpDefWrapper&, const UserOpConf&);
};

struct TmpSizeInferFnUtil {
  static size_t ZeroTmpSize(const InferContext&);
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_
