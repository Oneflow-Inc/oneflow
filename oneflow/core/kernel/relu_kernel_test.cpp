//#include "oneflow/core/kernel/relu_kernel.h"
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
//  FloatingPointType in_mat[8] = {1.0, -1.0, -2.0, 2.0, 0.0, 0.5, -10.0,
//  100.0}; FloatingPointType out_diff_mat[8] = {-8.0, 7.0, -6.0, 5.0,
//                                       -4.0, 3.0, -2.0, 1.0};
//  FloatingPointType out_mat[8] = {0};
//  FloatingPointType in_diff_mat[8] = {0};
//  FloatingPointType expected_out_mat[8] = {1.0, 0, 0, 2.0, 0, 0.5, 0, 100.0};
//  FloatingPointType expected_in_diff_mat[8] = {-8.0, 0, 0, 5.0, 0, 3.0,
//  0, 1.0}; auto bn2blob_ptr = new HashMap<std::string, Blob*>;
//  (*bn2blob_ptr)["in"] = KTCommon::CreateBlobWithVector({1, 8}, in_mat);
//  (*bn2blob_ptr)["out"] = KTCommon::CreateBlobWithVector({1, 8}, out_mat);
//  (*bn2blob_ptr)["in_diff"] =
//      KTCommon::CreateBlobWithVector({1, 8}, in_diff_mat);
//  (*bn2blob_ptr)["out_diff"] =
//      KTCommon::CreateBlobWithVector({1, 8}, out_diff_mat);
//  (*bn2blob_ptr)["expected_out"] =
//      KTCommon::CreateBlobWithVector({1, 8}, expected_out_mat);
//  (*bn2blob_ptr)["expected_in_diff"] =
//      KTCommon::CreateBlobWithVector({1, 8}, expected_in_diff_mat);
//  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
//}
//
// template<DeviceType device_type, typename FloatingPointType>
// Kernel* BuildReluKernel() {
//  OperatorConf op_conf;
//  op_conf.set_name("relu_op_test");
//  ReluOpConf* relu_conf = op_conf.mutable_relu_conf();
//  relu_conf->set_in("relu/in");
//  relu_conf->set_out("relu/out");
//  auto relu_op = ConstructOp(op_conf);
//  OperatorProto op_proto;
//  relu_op->ToProto(&op_proto);
//  auto relu_kernel = new ReluKernel<device_type, FloatingPointType>();
//  relu_kernel->InitFromOpProto(op_proto);
//  return relu_kernel;
//}
//
// template<DeviceType device_type, typename FloatingPointType>
// void TestReluKernel() {
//  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
//  KernelCtx ctx;
//  KTCommon::BuildKernelCtx(&ctx);
//  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, FloatingPointType>();
//  auto relu_kernel = BuildReluKernel<device_type, FloatingPointType>();
//  relu_kernel->Forward(ctx, BnInOp2BlobPtr);
//  relu_kernel->Backward(ctx, BnInOp2BlobPtr);
//  KTCommon::SyncStream(&ctx);
//  KTCommon::CheckResult(BnInOp2BlobPtr, "out", "expected_out");
//  KTCommon::CheckResult(BnInOp2BlobPtr, "in_diff", "expected_in_diff");
//}
//
//}  // namespace
//
//}  // namespace test
//
// TEST(ReluKernel, relu_kernel_cpu) {
//  test::TestReluKernel<DeviceType::kCPU, float>();
//  test::TestReluKernel<DeviceType::kCPU, double>();
//}
//
// TEST(ReluKernel, relu_kernel_gpu) {
//  test::TestReluKernel<DeviceType::kGPU, float>();
//  test::TestReluKernel<DeviceType::kGPU, double>();
//}
//
//}  // namespace oneflow
