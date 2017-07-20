#include "oneflow/core/kernel/softmax_loss_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename FloatingPointType>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  FloatingPointType in_mat[8] = {1, 2, 3, 4, 0, 0, 0, 0};
  FloatingPointType label_mat[2] = {2, 0};
  FloatingPointType expected_loss_mat[1] = {1.413242};
  FloatingPointType expected_in_diff_mat[8] = {
      0.0160293, 0.04357215, -0.3815586, 0.32195715,
      -0.375,    0.125,      0.125,      0.125};
  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
  (*bn2blob_ptr)["in"] = KTCommon::CreateBlobWithVector({2, 4}, in_mat);
  (*bn2blob_ptr)["label"] = KTCommon::CreateBlobWithVector({2}, label_mat);
  (*bn2blob_ptr)["prob"] = KTCommon::CreateBlobWithSameValue({2, 4}, 0.0);
  (*bn2blob_ptr)["tmp_1D"] = KTCommon::CreateBlobWithSameValue({2}, 0.0);
  (*bn2blob_ptr)["loss"] = KTCommon::CreateBlobWithSameValue({1}, 0.0);
  (*bn2blob_ptr)["in_diff"] = KTCommon::CreateBlobWithSameValue({2, 4}, 0.0);
  (*bn2blob_ptr)["expected_loss"] =
      KTCommon::CreateBlobWithVector({1}, expected_loss_mat);
  (*bn2blob_ptr)["expected_in_diff"] =
      KTCommon::CreateBlobWithVector({2, 4}, expected_in_diff_mat);
  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename FloatingPointType>
Kernel* BuildSoftmaxLossKernel() {
  OperatorConf op_conf;
  op_conf.set_name("softmax_loss_op_test");
  SoftmaxLossOpConf* softmax_loss_conf = op_conf.mutable_softmax_loss_conf();
  softmax_loss_conf->set_in("softmax_loss/in");
  softmax_loss_conf->set_label("softmax_loss/label");
  softmax_loss_conf->set_loss("softmax_loss/loss");
  auto softmax_loss_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  softmax_loss_op->ToProto(&op_proto);
  auto softmax_loss_kernel =
      new SoftmaxLossKernel<device_type, FloatingPointType>();
  softmax_loss_kernel->InitFromOpProto(op_proto);
  return softmax_loss_kernel;
}

template<DeviceType device_type, typename FloatingPointType>
void TestSoftmaxLossKernel() {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);
  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, FloatingPointType>();
  auto softmax_loss_kernel =
      BuildSoftmaxLossKernel<device_type, FloatingPointType>();
  softmax_loss_kernel->Forward(ctx, BnInOp2BlobPtr);
  auto loss = BnInOp2BlobPtr("loss")->dptr<FloatingPointType>();
  auto in_diff = BnInOp2BlobPtr("in_diff")->dptr<FloatingPointType>();
  auto prob = BnInOp2BlobPtr("prob")->dptr<FloatingPointType>();
  auto tmp = BnInOp2BlobPtr("tmp_1D")->dptr<FloatingPointType>();
  KTCommon::SyncStream(&ctx);
  KTCommon::CheckResult(BnInOp2BlobPtr, "loss", "expected_loss");
  KTCommon::CheckResult(BnInOp2BlobPtr, "in_diff", "expected_in_diff");
}

}  // namespace

}  // namespace test

TEST(SoftmaxLossKernel, softmax_loss_kernel_cpu) {
  test::TestSoftmaxLossKernel<DeviceType::kCPU, float>();
  test::TestSoftmaxLossKernel<DeviceType::kCPU, double>();
}

TEST(SoftmaxLossKernel, softmax_loss_kernel_gpu) {
  test::TestSoftmaxLossKernel<DeviceType::kGPU, float>();
  test::TestSoftmaxLossKernel<DeviceType::kGPU, double>();
}

}  // namespace oneflow
