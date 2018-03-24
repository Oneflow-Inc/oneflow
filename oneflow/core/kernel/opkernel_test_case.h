#ifndef ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_CASE_H_
#define ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_CASE_H_
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

namespace test {

class OpKernelTestCase final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernelTestCase);
  OpKernelTestCase();
  ~OpKernelTestCase() = default;

  void Run();

  //  Setters
  JobConf* mut_job_conf() { return &job_conf_; }
  void set_is_train(bool is_train);
  void set_device_type(DeviceType device_type) { device_type_ = device_type; }
  OperatorConf* mut_op_conf() { return &op_conf_; }
  void InitBlob(const std::string&, Blob* blob);
  void ForwardCheckBlob(const std::string&, DeviceType device_type, Blob* blob);
  void BackwardCheckBlob(const std::string&, DeviceType device_type,
                         Blob* blob);
  void set_is_forward(bool is_forward) { is_forward_ = is_forward; }

 private:
  std::function<Blob*(const std::string&)> MakeGetterBnInOp2Blob();
  std::function<BlobDesc*(const std::string&)> MakeGetterBnInOp2BlobDesc();
  void InitBeforeRun();
  void AssertAfterRun() const;

  HashMap<std::string, Blob*> bn_in_op2blob_;
  HashMap<std::string, BlobDesc> bn_in_op2blob_desc_;
  JobConf job_conf_;
  OperatorConf op_conf_;
  std::list<std::string> forward_asserted_blob_names_;
  std::list<std::string> backward_asserted_blob_names_;
  ParallelContext parallel_ctx_;
  KernelCtx kernel_ctx_;
  DeviceType device_type_;
  bool is_forward_;
};

}  // namespace test

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_CASE_H_
