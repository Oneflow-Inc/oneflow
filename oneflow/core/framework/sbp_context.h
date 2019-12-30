#ifndef ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_
#define ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_

#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/job/sbp_parallel.pb.h"

namespace oneflow {

namespace user_op {

class TensorDesc;

class SbpContext {
 public:
  virtual ~SbpContext() = default;

  virtual const TensorDesc& LogicalTensorDesc4InputArgNameAndIndex(
      const std::string& input_arg_name, int32_t index) = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;

  SbpSignatureList* sbp_sig_list() { return sbp_sig_list_; }

  template<typename T>
  T GetAttr(const std::string& attr_name) const {
    return user_op_conf_.attr<T>(attr_name);
  }

 protected:
  SbpContext(UserOpConfWrapper&& conf, SbpSignatureList* sbp_sig_list)
      : user_op_conf_(conf), sbp_sig_list_(sbp_sig_list) {}
  SbpContext(const SbpContext&) = delete;

 private:
  UserOpConfWrapper user_op_conf_;
  SbpSignatureList* sbp_sig_list_;
};

struct GetSbpFnUtil {
  static Maybe<void> MirrorSplitAtDim0(SbpContext*);
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_
