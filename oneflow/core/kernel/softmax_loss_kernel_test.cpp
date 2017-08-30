//#include "oneflow/core/kernel/softmax_loss_kernel.h"
//#include "oneflow/core/device/cpu_device_context.h"
//#include "oneflow/core/device/cuda_device_context.h"
//#include "oneflow/core/kernel/kernel_test_common.h"
//
// namespace oneflow {
//
// namespace test {
//
// namespace {
//
// template<DeviceType device_type, typename FloatingPointType>
// std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
//  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
//  FloatingPointType prediction_mat[8] = {1, 2, 3, 4, 0, 0, 0, 0};
//  FloatingPointType label_mat[2] = {2, 0};
//  FloatingPointType expected_loss_mat[1] = {2.826484};
//  FloatingPointType expected_prediction_diff_mat[8] = {
//      0.0320586, 0.0871443, -0.7631172, 0.6439143, -0.75, 0.25, 0.25, 0.25};
//  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
//  (*bn2blob_ptr)["prediction"] =
//      KTCommon::CreateBlobWithVector({2, 4}, prediction_mat);
//  (*bn2blob_ptr)["label"] = KTCommon::CreateBlobWithVector({2}, label_mat);
//  (*bn2blob_ptr)["prob"] = KTCommon::CreateBlobWithRandomValue({2, 4});
//  (*bn2blob_ptr)["tmp_1D"] = KTCommon::CreateBlobWithRandomValue({2});
//  (*bn2blob_ptr)["loss"] = KTCommon::CreateBlobWithRandomValue({1});
//  (*bn2blob_ptr)["prediction_diff"] =
//      KTCommon::CreateBlobWithRandomValue({2, 4});
//  (*bn2blob_ptr)["expected_loss"] =
//      KTCommon::CreateBlobWithVector({1}, expected_loss_mat);
//  (*bn2blob_ptr)["expected_prediction_diff"] =
//      KTCommon::CreateBlobWithVector({2, 4}, expected_prediction_diff_mat);
//  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
//}
//
// template<DeviceType device_type, typename FloatingPointType>
// Kernel* BuildSoftmaxLossKernel() {
//  OperatorConf op_conf;
//  op_conf.set_name("softmax_loss_op_test");
//  SoftmaxLossOpConf* softmax_loss_conf = op_conf.mutable_softmax_loss_conf();
//  softmax_loss_conf->set_prediction("softmax_loss/prediction");
//  softmax_loss_conf->set_label("softmax_loss/label");
//  softmax_loss_conf->set_loss("softmax_loss/loss");
//  auto softmax_loss_op = ConstructOp(op_conf);
//  OperatorProto op_proto;
//  softmax_loss_op->ToProto(&op_proto);
//  auto softmax_loss_kernel =
//      new SoftmaxLossKernel<device_type, FloatingPointType>();
//  softmax_loss_kernel->InitFromOpProto(op_proto);
//  return softmax_loss_kernel;
//}
//
// template<DeviceType device_type, typename FloatingPointType>
// void TestSoftmaxLossKernel() {
//  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
//  KernelCtx ctx;
//  KTCommon::BuildKernelCtx(&ctx);
//  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, FloatingPointType>();
//  auto softmax_loss_kernel =
//      BuildSoftmaxLossKernel<device_type, FloatingPointType>();
//  softmax_loss_kernel->Forward(ctx, BnInOp2BlobPtr);
//  KTCommon::SyncStream(&ctx);
//  KTCommon::CheckResult(BnInOp2BlobPtr, "loss", "expected_loss");
//  KTCommon::CheckResult(BnInOp2BlobPtr, "prediction_diff",
//                        "expected_prediction_diff");
//}
//
//}  // namespace
//
//}  // namespace test
//
// TEST(SoftmaxLossKernel, softmax_loss_kernel_cpu) {
//  test::TestSoftmaxLossKernel<DeviceType::kCPU, float>();
//  test::TestSoftmaxLossKernel<DeviceType::kCPU, double>();
//}
//
// TEST(SoftmaxLossKernel, softmax_loss_kernel_gpu) {
//  test::TestSoftmaxLossKernel<DeviceType::kGPU, float>();
//  test::TestSoftmaxLossKernel<DeviceType::kGPU, double>();
//}
//
//}  // namespace oneflow
