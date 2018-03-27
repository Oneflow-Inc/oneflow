#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/kernel/opkernel_test_common.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace test {

namespace {

std::string ExpectedBlobName(const std::string& name) {
  return name + "_$expected$";
}

void BlobCmp(const std::string& blob_name, DeviceType device_type,
             const Blob* lhs, const Blob* rhs) {
  DataType data_type = lhs->data_type();

#define BLOB_CMP_ENTRY(dev_type, data_type_pair)                             \
  if (device_type == dev_type                                                \
      && data_type == OF_PP_PAIR_SECOND(data_type_pair)) {                   \
    KTCommon<dev_type, OF_PP_PAIR_FIRST(data_type_pair)>::BlobCmp(blob_name, \
                                                                  lhs, rhs); \
    return;                                                                  \
  }
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(BLOB_CMP_ENTRY, DEVICE_TYPE_SEQ,
                                   ALL_DATA_TYPE_SEQ);

  UNIMPLEMENTED();
}

Blob* CreateBlobWithRandomVal(DeviceType device_type,
                              const BlobDesc* blob_desc) {
#define CREATE_BLOB_ENTRY(dev_type, data_type_pair)                     \
  if (device_type == dev_type                                           \
      && blob_desc->data_type() == OF_PP_PAIR_SECOND(data_type_pair)) { \
    return KTCommon<dev_type, OF_PP_PAIR_FIRST(data_type_pair)>::       \
        CreateBlobWithRandomVal(blob_desc);                             \
  }
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(CREATE_BLOB_ENTRY, DEVICE_TYPE_SEQ,
                                   ALL_DATA_TYPE_SEQ);
  UNIMPLEMENTED();
  return nullptr;
}

}  // namespace

void OpKernelTestCase::InitBlob(const std::string& name, Blob* blob) {
  CHECK(bn_in_op2blob_.emplace(name, blob).second);
}

void OpKernelTestCase::ForwardCheckBlob(const std::string& name,
                                        DeviceType device_type, Blob* blob,
                                        bool need_random_init) {
  forward_asserted_blob_names_.push_back(name);
  if (need_random_init) {
    InitBlob(name, CreateBlobWithRandomVal(device_type, blob->blob_desc_ptr()));
  }
  CHECK(bn_in_op2blob_.emplace(ExpectedBlobName(name), blob).second);
}

void OpKernelTestCase::ForwardCheckBlob(const std::string& name,
                                        DeviceType device_type, Blob* blob) {
  ForwardCheckBlob(name, device_type, blob, true);
}

void OpKernelTestCase::BackwardCheckBlob(const std::string& name,
                                         DeviceType device_type, Blob* blob,
                                         bool need_random_init) {
  backward_asserted_blob_names_.push_back(name);
  if (need_random_init) {
    InitBlob(name, CreateBlobWithRandomVal(device_type, blob->blob_desc_ptr()));
  }
  CHECK(bn_in_op2blob_.emplace(ExpectedBlobName(name), blob).second);
}

void OpKernelTestCase::BackwardCheckBlob(const std::string& name,
                                         DeviceType device_type, Blob* blob) {
  BackwardCheckBlob(name, device_type, blob, true);
}

std::function<Blob*(const std::string&)>
OpKernelTestCase::MakeGetterBnInOp2Blob() {
  return [this](const std::string& bn_in_op) {
    if (bn_in_op2blob_[bn_in_op] == nullptr) {
      bn_in_op2blob_[bn_in_op] = CreateBlobWithRandomVal(
          device_type_, &bn_in_op2blob_desc_.at(bn_in_op));
    }
    return bn_in_op2blob_.at(bn_in_op);
  };
}

OpKernelTestCase::OpKernelTestCase() {
  parallel_ctx_.set_parallel_id(0);
  parallel_ctx_.set_parallel_num(1);
  parallel_ctx_.set_policy(ParallelPolicy::kModelParallel);
}

void OpKernelTestCase::InitBeforeRun() {
  JobDescProto job_desc_proto;
  *job_desc_proto.mutable_job_conf() = job_conf_;
  if (Global<JobDesc>::Get()) { Global<JobDesc>::Delete(); }
  Global<JobDesc>::New(job_desc_proto);

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

void OpKernelTestCase::set_is_train(bool is_train) {
  if (is_train) {
    mut_job_conf()->mutable_train_conf();
  } else {
    mut_job_conf()->mutable_predict_conf();
  }
}

void OpKernelTestCase::AssertAfterRun() const {
  const std::list<std::string>* asserted_blob_names = nullptr;
  if (Global<JobDesc>::Get()->IsPredict() || is_forward_) {
    asserted_blob_names = &forward_asserted_blob_names_;
  } else {
    asserted_blob_names = &backward_asserted_blob_names_;
  }
  for (const auto& blob_name : *asserted_blob_names) {
    BlobCmp(blob_name, device_type_, bn_in_op2blob_.at(blob_name),
            bn_in_op2blob_.at(ExpectedBlobName(blob_name)));
  }
}

void OpKernelTestCase::Run() {
  InitBeforeRun();
  auto op = ConstructOp(op_conf_);
  auto BnInOp2BlobDesc = MakeGetterBnInOp2BlobDesc();
  op->InferBlobDescs(BnInOp2BlobDesc, &parallel_ctx_, device_type_);
  std::list<bool> is_forward_launch_types;
  if (Global<JobDesc>::Get()->IsPredict() || is_forward_) {
    is_forward_launch_types = {true};
  } else {
    is_forward_launch_types = {true, false};
  }
  auto BnInOp2Blob = MakeGetterBnInOp2Blob();
  for (bool is_forward : is_forward_launch_types) {
    KernelConf kernel_conf;
    op->GenKernelConf(BnInOp2BlobDesc, is_forward, device_type_, &parallel_ctx_,
                      &kernel_conf);
    auto kernel = ConstructKernel(&parallel_ctx_, kernel_conf);
    kernel->Launch(kernel_ctx_, BnInOp2Blob);
  }
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
