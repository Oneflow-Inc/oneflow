#ifndef ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_
#define ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_

#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/job/sbp_parallel.pb.h"

namespace oneflow {

namespace user_op {

class TensorDesc;

class UserOpSbpSignatureBuilder final {
 public:
  UserOpSbpSignatureBuilder(SbpSignatureList* sbp_sig_list) : sbp_sig_list_(sbp_sig_list) {}

  UserOpSbpSignatureBuilder& Split(const OpArg& op_arg, int64_t axis);
  UserOpSbpSignatureBuilder& Split(const std::vector<OpArg>& op_args, int64_t axis);
  UserOpSbpSignatureBuilder& Split(const std::vector<std::pair<std::string, int32_t>>& op_args,
                                   int64_t axis);

  UserOpSbpSignatureBuilder& Broadcast(const OpArg& op_arg);
  UserOpSbpSignatureBuilder& Broadcast(const std::vector<OpArg>& op_args);
  UserOpSbpSignatureBuilder& Broadcast(const std::vector<std::pair<std::string, int32_t>>& op_args);

  UserOpSbpSignatureBuilder& PartialSum(const OpArg& op_arg);
  UserOpSbpSignatureBuilder& PartialSum(const std::vector<OpArg>& op_args);
  UserOpSbpSignatureBuilder& PartialSum(
      const std::vector<std::pair<std::string, int32_t>>& op_args);

  void Build() { *(sbp_sig_list_->mutable_sbp_signature()->Add()) = sbp_sig_tmp_; }

 private:
  SbpSignatureList* sbp_sig_list_;
  SbpSignature sbp_sig_tmp_;
};

class SbpContext {
 public:
  virtual ~SbpContext() = default;

  virtual const TensorDesc& LogicalTensorDesc4InputArgNameAndIndex(
      const std::string& input_arg_name, int32_t index) const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;

  UserOpSbpSignatureBuilder NewBuilder() { return UserOpSbpSignatureBuilder(sbp_sig_list_); }

  DeviceType device_type() const { return device_type_; }
  int64_t parallel_num() const { return parallel_num_; }

  template<typename T>
  T GetAttr(const std::string& attr_name) const {
    return user_op_conf_.attr<T>(attr_name);
  }

 protected:
  SbpContext(UserOpConfWrapper&& conf, SbpSignatureList* sbp_sig_list, DeviceType device_type,
             int64_t parallel_num)
      : user_op_conf_(conf),
        sbp_sig_list_(sbp_sig_list),
        device_type_(device_type),
        parallel_num_(parallel_num) {}
  SbpContext(const SbpContext&) = delete;

 private:
  UserOpConfWrapper user_op_conf_;
  SbpSignatureList* sbp_sig_list_;
  DeviceType device_type_;
  int64_t parallel_num_;
};

struct GetSbpFnUtil {
  static Maybe<void> DefaultBroadcastToBroadcast(SbpContext*);
  static Maybe<void> SplitForEachAxis(SbpContext*);
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_
