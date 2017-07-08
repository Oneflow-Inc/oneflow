#include "oneflow/core/kernel/model_update_kernel.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename FloatingPointType>
Kernel* BuildMdUpdateKernel(float learn_rate) {
  OperatorConf op_conf;
  op_conf.set_name("model_update_test");
  ModelUpdateOpConf* model_update_conf = op_conf.mutable_model_update_conf();
  model_update_conf->set_learn_rate(learn_rate);
  auto model_update_op = OpMgr::Singleton()->ConstructOp(op_conf);
  OperatorProto op_proto;
  model_update_op->ToProto(&op_proto);
  auto model_update_kernel =
      new MdUpdateKernel<device_type, FloatingPointType>();
  model_update_kernel->InitFromOpProto(op_proto);
  return model_update_kernel;
}

template<DeviceType device_type, typename FloatingPointType>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;

  std::vector<int64_t> dim_vec = {1, 3, 2};

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
  (*bn2blob_ptr)["model"] = KTCommon::CreateBlobWithSameValue(dim_vec, 3);
  (*bn2blob_ptr)["model_diffs"] = KTCommon::CreateBlobWithSameValue(dim_vec, 2);
  (*bn2blob_ptr)["model_expected"] =
      KTCommon::CreateBlobWithSameValue(dim_vec, 1);
  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename FloatingPointType>
void TestMdUpdateKernel() {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);

  const float learn_rate = {1.0f};
  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, FloatingPointType>();
  auto model_update_kernel =
      BuildMdUpdateKernel<device_type, FloatingPointType>(learn_rate);

  model_update_kernel->Backward(ctx, BnInOp2BlobPtr);
  KTCommon::SyncStream(&ctx);

  KTCommon::CheckResult(BnInOp2BlobPtr, "model", "model_expected");
}

}  // namespace

}  // namespace test

TEST(MdUpdateKernel, model_update_cpu) {
  test::TestMdUpdateKernel<DeviceType::kCPU, float>();
  test::TestMdUpdateKernel<DeviceType::kCPU, double>();
}

TEST(MdUpdateKernel, model_update_gpu) {
  test::TestMdUpdateKernel<DeviceType::kGPU, float>();
  test::TestMdUpdateKernel<DeviceType::kGPU, double>();
}

}  // namespace oneflow
