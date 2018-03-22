#ifndef ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_CASE_H_
#define ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_CASE_H_
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace test {

class OpKernelTestCase;

class OpKernelTestCaseBuilder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernelTestCaseBuilder);
  explicit OpKernelTestCaseBuilder(OpKernelTestCase* opkernel_test_case)
    : opkernel_test_case_(opkernel_test_case) {}

  JobConfProto* mut_job_conf_proto() {
    return opkernel_test_case_->mut_job_conf_proto();
  }
  OperatorConf* mut_op_conf() { return opkernel_test_case_->mut_op_conf(); }

  void InitBlob(const std::string&, std::unique_ptr<Blob>&& blob);
  void ForwardAssertEqBlob(const std::string&, std::unique_ptr<Blob>&& blob);
  void BackwardAssertEqBlob(const std::string&, std::unique_ptr<Blob>&& blob);
  
 private:
  HashMap<std::string, std::unique_ptr<Blob>>* mut_bn_in_op2blob() {
    return opkernel_test_case_->mut_bn_in_op2blob();
  }
  OpKernelTestCase* opkernel_test_case_;
};

class OpKernelTestCase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernelTestCase);
  OpKernelTestCase()
    : bn_in_op2blob_(new HashMap<std::string, std::unique_ptr<Blob>>()),
      bn_in_op2blob_desc_(new HashMap<std::string, std::unique_ptr<BlobDesc*>>()) {}
  ~OpKernelTestCase() = default;

  virtual void Run() const = 0;

  void Build();
  virtual void Build(OpKernelTestCaseBuilder* builder) = 0;

  std::function<Blob*(const std::string&)> MakeGetterBnInOp2Blob() const;
  std::function<BlobDesc*(const std::string&)> MakeGetterBnInOp2BlobDesc() const;
  const JobConfProto& job_conf_proto() const { return job_conf_proto_; }
  const OperatorConf& op_conf() const { return op_conf_; }
  const std::list<std::string>& forward_asserted_blob() const {
    return forward_asserted_blob_;
  }
  const std::list<std::string>& backward_asserted_blob() const {
    return backward_asserted_blob_;
  }
  
  HashMap<std::string, std::unique_ptr<Blob>>* mut_bn_in_op2blob() {
    return bn_in_op2blob_.get();
  }
  HashMap<std::string, std::unique_ptr<BlobDesc>>* mut_bn_in_op2blob_desc() {
    return bn_in_op2blob_desc_.get();
  }
  JobConfProto* mut_job_conf_proto() { return &job_conf_proto_; }
  OperatorConf* mut_op_conf() { return &op_conf_; }
  std::list<std::string>* mut_forward_asserted_blob() {
    return &forward_asserted_blob_;
  }
  std::list<std::string>* mut_backward_asserted_blob() {
    return &backward_asserted_blob_;
  }
  
 private:
  std::shared_ptr<HashMap<std::string, std::unique_ptr<Blob>>> bn_in_op2blob_;
  std::shared_ptr<HashMap<std::string, std::unique_ptr<BlobDesc*>>>
    bn_in_op2blob_desc_;
  JobConfProto job_conf_proto_;
  OperatorConf op_conf_;
  std::list<std::string> forward_asserted_blob_;
  std::list<std::string> backward_asserted_blob_;
  std::string model_save_dir_;
};

}

}

#endif  // ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_CASE_H_
