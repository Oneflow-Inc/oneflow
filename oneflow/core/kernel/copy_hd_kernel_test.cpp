//#include "oneflow/core/kernel/copy_hd_kernel.h"
//#include "oneflow/core/device/cuda_device_context.h"
//#include "oneflow/core/kernel/kernel_test_common.h"
//
// namespace oneflow {
//
// namespace test {
//
// namespace {
//
// template<typename FloatingPointType>
// std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr(
//    CopyHdOpConf::Type hd_type) {
//  using KTCommonCpu = KernelTestCommon<DeviceType::kCPU, FloatingPointType>;
//  using KTCommonGpu = KernelTestCommon<DeviceType::kGPU, FloatingPointType>;
//
//  std::vector<int64_t> dim_vec = {3, 4, 5, 6};
//
//  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
//
//  if (hd_type == CopyHdOpConf::H2D) {
//    (*bn2blob_ptr)["in"] = KTCommonCpu::CreateBlobWithSameValue(dim_vec, 1);
//    (*bn2blob_ptr)["out"] = KTCommonGpu::CreateBlobWithSameValue(dim_vec, 2);
//    (*bn2blob_ptr)["in_diff"] =
//        KTCommonCpu::CreateBlobWithSameValue(dim_vec, 3);
//    (*bn2blob_ptr)["out_diff"] = (*bn2blob_ptr)["out"];
//  } else {
//    (*bn2blob_ptr)["in"] = KTCommonGpu::CreateBlobWithSameValue(dim_vec, 1);
//    (*bn2blob_ptr)["out"] = KTCommonCpu::CreateBlobWithSameValue(dim_vec, 2);
//    (*bn2blob_ptr)["in_diff"] =
//        KTCommonGpu::CreateBlobWithSameValue(dim_vec, 3);
//    (*bn2blob_ptr)["out_diff"] = (*bn2blob_ptr)["out"];
//  }
//  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
//}
//
// template<typename FloatingPointType>
// Kernel* BuildCopyHdKernel(CopyHdOpConf::Type hd_type) {
//  OperatorConf op_conf;
//  op_conf.set_name("copy_hd_test");
//  CopyHdOpConf* copy_hd_conf = op_conf.mutable_copy_hd_conf();
//  copy_hd_conf->set_type(hd_type);
//  auto copy_hd_op = ConstructOp(op_conf);
//
//  OperatorProto op_proto;
//  copy_hd_op->ToProto(&op_proto);
//  auto copy_hd_kernel = new CopyHdKernel<kGPU, FloatingPointType>();
//  copy_hd_kernel->InitFromOpProto(op_proto);
//  return copy_hd_kernel;
//}
//
// template<typename FloatingPointType>
// void TestCopyHdKernel(CopyHdOpConf::Type hd_type) {
//  using KTCommonCpu = KernelTestCommon<DeviceType::kCPU, FloatingPointType>;
//  using KTCommonGpu = KernelTestCommon<DeviceType::kGPU, FloatingPointType>;
//
//  KernelCtx ctx;
//  KTCommonGpu::BuildKernelCtx(&ctx);
//
//  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<FloatingPointType>(hd_type);
//
//  auto copy_hd_kernel = BuildCopyHdKernel<FloatingPointType>(hd_type);
//
//  copy_hd_kernel->Forward(ctx, BnInOp2BlobPtr);
//  copy_hd_kernel->Backward(ctx, BnInOp2BlobPtr);
//  KTCommonGpu::SyncStream(&ctx);
//
//  if (hd_type == CopyHdOpConf::H2D) {
//    KTCommonCpu::CheckResult(BnInOp2BlobPtr, "in", "in_diff");
//  } else {
//    KTCommonGpu::CheckResult(BnInOp2BlobPtr, "in", "in_diff");
//  }
//}
//
//}  // namespace
//
//}  // namespace test
//
// TEST(CopyHdKernel, copy_d2h) {
//  test::TestCopyHdKernel<float>(CopyHdOpConf::D2H);
//  test::TestCopyHdKernel<double>(CopyHdOpConf::D2H);
//}
//
// TEST(CopyHdKernel, copy_h2d) {
//  test::TestCopyHdKernel<float>(CopyHdOpConf::H2D);
//  test::TestCopyHdKernel<double>(CopyHdOpConf::H2D);
//}
//
//}  // namespace oneflow
