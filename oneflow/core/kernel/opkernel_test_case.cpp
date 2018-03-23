#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/kernel/opkernel_test_common.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace test {

namespace {

std::string ExpectedBlobName(const std::string& name) {
  return name + "_$expected$";
}

void BlobCmp(DeviceType device_type, const Blob* lhs, const Blob* rhs) {
  DataType data_type = lhs->data_type();

#define BLOB_CMP_ENTRY(dev_type, data_type_pair)                             \
  if (device_type == dev_type                                                \
      && data_type == OF_PP_PAIR_SECOND(data_type_pair)) {                   \
    KTCommon<dev_type, OF_PP_PAIR_FIRST(data_type_pair)>::BlobCmp(lhs, rhs); \
    return;                                                                  \
  }
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(BLOB_CMP_ENTRY, DEVICE_TYPE_SEQ,
                                   ALL_DATA_TYPE_SEQ);

  UNIMPLEMENTED();
}

}  // namespace

JobConf* OpKernelTestCaseBuilder::mut_job_conf_proto() {
  return opkernel_test_case_->mut_job_conf_proto();
}
OperatorConf* OpKernelTestCaseBuilder::mut_op_conf() {
  return opkernel_test_case_->mut_op_conf();
}
ParallelContext* OpKernelTestCaseBuilder::mut_parallel_ctx() {
  return opkernel_test_case_->mut_parallel_ctx();
}

void OpKernelTestCaseBuilder::set_device_type(DeviceType device_type) {
  opkernel_test_case_->set_device_type(device_type);
}

void OpKernelTestCaseBuilder::set_is_forward(bool is_forward) {
  opkernel_test_case_->set_is_forward(is_forward);
}

HashMap<std::string, Blob*>* OpKernelTestCaseBuilder::mut_bn_in_op2blob() {
  return opkernel_test_case_->mut_bn_in_op2blob();
}

void OpKernelTestCaseBuilder::InitBlob(const std::string& name, Blob* blob) {
  CHECK(mut_bn_in_op2blob()->emplace(name, blob).second);
}

void OpKernelTestCaseBuilder::ForwardAssertEqBlob(const std::string& name,
                                                  Blob* blob) {
  opkernel_test_case_->mut_forward_asserted_blob_names()->push_back(name);
  CHECK(mut_bn_in_op2blob()->emplace(ExpectedBlobName(name), blob).second);
}

void OpKernelTestCaseBuilder::BackwardAssertEqBlob(const std::string& name,
                                                   Blob* blob) {
  opkernel_test_case_->mut_backward_asserted_blob_names()->push_back(name);
  CHECK(mut_bn_in_op2blob()->emplace(ExpectedBlobName(name), blob).second);
}

std::function<Blob*(const std::string&)>
OpKernelTestCase::MakeGetterBnInOp2Blob() {
  return [this](const std::string& bn_in_op) {
    bn_in_op2blob_[bn_in_op] =
        NewBlob(nullptr, nullptr, nullptr, nullptr, DeviceType::kCPU);
    return bn_in_op2blob_.at(bn_in_op);
  };
}

void OpKernelTestCase::InitBeforeRun() {
  JobDescProto job_desc_proto;
  *job_desc_proto.mutable_job_conf() = job_conf_proto_;
  JobDesc::DeleteSingleton();
  JobDesc::NewSingleton(job_desc_proto);

  parallel_ctx_.set_parallel_id(0);
  parallel_ctx_.set_parallel_num(1);
  parallel_ctx_.set_policy(ParallelPolicy::kModelParallel);

  for (const auto& pair : bn_in_op2blob_) {
    bn_in_op2blob_desc_[pair.first] = pair.second->blob_desc();
  }

  if (device_type_ == DeviceType::kCPU) {
    BuildKernelCtx<DeviceType::kCPU>(&kernel_ctx_);
  } else if (device_type_ == DeviceType::kGPU) {
    BuildKernelCtx<DeviceType::kGPU>(&kernel_ctx_);
  } else {
    UNIMPLEMENTED();
  }
}

void OpKernelTestCase::AssertAfterRun() const {
  const std::list<std::string>* asserted_blob_names = nullptr;
  if (JobDesc::Singleton()->IsPredict() || is_forward_) {
    asserted_blob_names = &forward_asserted_blob_names_;
  } else {
    asserted_blob_names = &backward_asserted_blob_names_;
  }

  for (const auto& blob_name : *asserted_blob_names) {
    BlobCmp(device_type_, bn_in_op2blob_.at(blob_name),
            bn_in_op2blob_.at(ExpectedBlobName(blob_name)));
  }
}

void OpKernelTestCase::Run() {
  InitBeforeRun();
  if (JobDesc::Singleton()->IsPredict() && !is_forward_) { return; }
  auto op = ConstructOp(op_conf_);
  auto BnInOp2BlobDesc = MakeGetterBnInOp2BlobDesc();
  KernelConf kernel_conf;
  op->InferBlobDescs(BnInOp2BlobDesc, &parallel_ctx_, device_type_);
  op->GenKernelConf(BnInOp2BlobDesc, is_forward_, device_type_, &parallel_ctx_,
                    &kernel_conf);
  auto kernel = ConstructKernel(&parallel_ctx_, kernel_conf);
  kernel->Launch(kernel_ctx_, MakeGetterBnInOp2Blob());
  AssertAfterRun();
}

std::function<BlobDesc*(const std::string&)>
OpKernelTestCase::MakeGetterBnInOp2BlobDesc() {
  return [this](const std::string& bn_in_op) {
    return &bn_in_op2blob_desc_[bn_in_op];
  };
}

}  // namespace test

}  // namespace oneflow
