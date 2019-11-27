#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_CONF_BUILDER_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_CONF_BUILDER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/blob_info.h"
#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

class BlobDesc;

namespace user_op {

class UserOpConfWrapper final {
 public:
  UserOpConfWrapper(const OperatorConf&);
  std::string op_name() const;
  std::string op_type_name() const;
  std::string input(const std::string& arg_name, int32_t index) const;
  std::string output(const std::string& arg_name, int32_t index) const;

  template<typename T>
  T attr(const std::string& attr_name) const;

 private:
  UserOpConfWrapper() = default;
  friend class UserOpConfWrapperBuilder;

  OperatorConf op_conf_;
};

class UserOpWrapper final {
 public:
  UserOpWrapper(const OperatorConf& op,
                const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp);

  const BlobInfo& LogicalBlobInfo4ArgNameAndIndex(const std::string& arg_name, int32_t index) const;
  const UserOpConfWrapper& user_op_conf_wrapper();
  std::string op_name() const { return conf_.op_name(); }
  std::string op_type_name() const { return conf_.op_type_name(); }
  std::string input(const std::string& arg_name, int32_t index) const {
    return conf_.input(arg_name, index);
  }
  std::string output(const std::string& arg_name, int32_t index) const {
    return conf_.output(arg_name, index);
  }
  template<typename T>
  T attr(const std::string& attr_name) const {
    return conf_.attr<T>(attr_name);
  }

 private:
  UserOpConfWrapper conf_;
  HashMap<std::string, BlobInfo> bn2blob_info_;
};

class UserOpConfWrapperBuilder final {
 public:
  UserOpConfWrapperBuilder(const std::string& op_name) : op_name_(op_name) {}
  UserOpConfWrapperBuilder& Op(const std::string& op_type_name) {
    op_type_name_ = op_type_name;
    return *this;
  }
  UserOpConfWrapperBuilder& Input(const std::string& arg_name,
                                  const std::string& logical_blob_name);
  UserOpConfWrapperBuilder& Output(const std::string& arg_name, int32_t num);
  template<typename T>
  UserOpConfWrapperBuilder& Attr(const std::string& attr_name, const T& val);

  UserOpConfWrapper Build();

 private:
  UserOpConfWrapper wrapper_;
  std::string op_name_;
  std::string op_type_name_;
  HashMap<std::string, std::vector<std::string>> input_;
  HashMap<std::string, std::vector<std::string>> output_;
  HashMap<std::string, UserOpAttrVal> attr_;
};

}  // namespace user_op

Maybe<OperatorConf> CheckAndCompleteUserOpConf(const OperatorConf& op_conf);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_CONF_BUILDER_H_
