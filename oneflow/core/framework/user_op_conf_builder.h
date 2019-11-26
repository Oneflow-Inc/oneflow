#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_CONF_BUILDER_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_CONF_BUILDER_H_

#include "oneflow/core/common/util.h"
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
  OperatorConf op_conf_;
};

class UserOpWrapper final {
 public:
  UserOpWrapper(const OperatorConf& op, const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp);

  const BlobInfo& LogicalBlobInfo4ArgNameAndIndex(const std::string& arg_name, int32_t index) const;
  std::string LogicalBlobName4ArgNameAndIndex(const std::string& arg_name, int32_t index) const;

  const UserOpConfWrapper& op_conf();
  std::string op_name() const;
  std::string op_type_name() const;
  std::string input(const std::string& arg_name, int32_t index) const;
  std::string output(const std::string& arg_name, int32_t index) const;

  template<typename T>
  T attr(const std::string& attr_name) const;
 private: 
  UserOpConfWrapper user_op_conf_wrapper_;
  HashMap<std::string, BlobInfo> bn2blob_info_;
}

class UserOpConfWrapperBuilder final {
 public:
  UserOpConfWrapperBuilder(const std::string& op_name);
  UserOpConfWrapperBuilder& Input(const std::string& arg_name, const std::string& logical_blob_name);
  UserOpConfWrapperBuilder& Output(const std::string& arg_name, const std::string& logical_blob_name);
  template<typename T>
  UserOpConfWrapperBuilder& Attr(const std::string& attr_name, const T& val);

  UserOpConfWrapper Build();
 private:
  UserOpConfWrapper wrapper_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_CONF_BUILDER_H_
