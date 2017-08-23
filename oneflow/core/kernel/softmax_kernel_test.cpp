//#include "oneflow/core/kernel/softmax_kernel.h"
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
//  FloatingPointType in_mat[8] = {1, 2, 3, 4, 0, 0, 0, 0};
//  FloatingPointType out_diff_mat[8] = {0.2, 1, 2, 3, -4.0, 3.0, -2.0, 1.0};
//  FloatingPointType expected_out_mat[8] = {
//      0.0320586, 0.0871443, 0.2368828, 0.6439143, 0.25, 0.25, 0.25, 0.25};
//  FloatingPointType expected_in_diff_mat[8] = {
//      -0.0737048, -0.1306350, -0.1182198, 0.3225595,
//      -0.875,     0.875,      -0.375,     0.375};
//  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
//  (*bn2blob_ptr)["in"] = KTCommon::CreateBlobWithVector({2, 4}, in_mat);
//  (*bn2blob_ptr)["out"] = KTCommon::CreateBlobWithSameValue({2, 4}, 0.0);
//  (*bn2blob_ptr)["tmp"] = KTCommon::CreateBlobWithSameValue({2}, 0.0);
//  (*bn2blob_ptr)["in_diff"] = KTCommon::CreateBlobWithSameValue({2, 4}, 0.0);
//  (*bn2blob_ptr)["out_diff"] =
//      KTCommon::CreateBlobWithVector({2, 4}, out_diff_mat);
//  (*bn2blob_ptr)["expected_out"] =
//      KTCommon::CreateBlobWithVector({2, 4}, expected_out_mat);
//  (*bn2blob_ptr)["expected_in_diff"] =
//      KTCommon::CreateBlobWithVector({2, 4}, expected_in_diff_mat);
//  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
//}
//
// template<DeviceType device_type, typename FloatingPointType>
// Kernel* BuildSoftmaxKernel() {
//  OperatorConf op_conf;
//  op_conf.set_name("softmax_op_test");
//  SoftmaxOpConf* softmax_conf = op_conf.mutable_softmax_conf();
//  softmax_conf->set_in("softmax/in");
//  softmax_conf->set_out("softmax/out");
//  auto softmax_op = ConstructOp(op_conf);
//  OperatorProto op_proto;
//  softmax_op->ToProto(&op_proto);
//  auto softmax_kernel = new SoftmaxKernel<device_type, FloatingPointType>();
//  softmax_kernel->InitFromOpProto(op_proto);
//  return softmax_kernel;
//}
//
// template<DeviceType device_type, typename FloatingPointType>
// void TestSoftmaxKernel() {
//  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
//  KernelCtx ctx;
//  KTCommon::BuildKernelCtx(&ctx);
//  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, FloatingPointType>();
//  auto softmax_kernel = BuildSoftmaxKernel<device_type, FloatingPointType>();
//  softmax_kernel->Forward(ctx, BnInOp2BlobPtr);
//  softmax_kernel->Backward(ctx, BnInOp2BlobPtr);
//  KTCommon::SyncStream(&ctx);
//  KTCommon::CheckResult(BnInOp2BlobPtr, "out", "expected_out");
//  KTCommon::CheckResult(BnInOp2BlobPtr, "in_diff", "expected_in_diff");
//}
//
//}  // namespace
//
//}  // namespace test
//
// TEST(SoftmaxKernel, softmax_kernel_cpu) {
//  test::TestSoftmaxKernel<DeviceType::kCPU, float>();
//  test::TestSoftmaxKernel<DeviceType::kCPU, double>();
//}
//
// TEST(SoftmaxKernel, softmax_kernel_gpu) {
//  test::TestSoftmaxKernel<DeviceType::kGPU, float>();
//  test::TestSoftmaxKernel<DeviceType::kGPU, double>();
//}
//
//}  // namespace oneflow
