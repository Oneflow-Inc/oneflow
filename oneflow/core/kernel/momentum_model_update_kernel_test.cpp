//#include "oneflow/core/kernel/momentum_model_update_kernel.h"
//#include "oneflow/core/kernel/kernel_test_common.h"
//
// namespace oneflow {
//
// namespace test {
//
// namespace {
//
// template<DeviceType device_type, typename FloatingPointType>
// Kernel* BuildMomentumMdUpdateKernel(float learning_rate, float beta) {
//  OperatorConf op_conf;
//  op_conf.set_name("momentum_model_update_test");
//  MomentumModelUpdateOpConf* momentum_md_update_conf =
//      op_conf.mutable_momentum_mdupdt_conf();
//  momentum_md_update_conf->set_learning_rate(learning_rate);
//  momentum_md_update_conf->set_beta(beta);
//  auto momentum_md_update_op = ConstructOp(op_conf);
//  OperatorProto op_proto;
//  momentum_md_update_op->ToProto(&op_proto);
//  auto momentum_md_update_kernel =
//      new MomentumMdUpdateKernel<device_type, FloatingPointType>();
//  momentum_md_update_kernel->InitFromOpProto(op_proto);
//  return momentum_md_update_kernel;
//}
//
// void InitJobDesc(int32_t piece_size, int32_t num_of_pieces_in_batch) {
//  JobConf job_conf;
//  job_conf.set_piece_size(piece_size);
//  job_conf.set_num_of_pieces_in_batch(num_of_pieces_in_batch);
//  JobDesc::Singleton()->InitFromJobConf(job_conf);
//}
//
// template<DeviceType device_type, typename FloatingPointType>
// std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
//  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
//
//  std::vector<int64_t> dim_vec = {1, 3, 2};
//
//  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
//  (*bn2blob_ptr)["model"] = KTCommon::CreateBlobWithSameValue(dim_vec, 2);
//  (*bn2blob_ptr)["momentum"] = KTCommon::CreateBlobWithSameValue(dim_vec, 4);
//  (*bn2blob_ptr)["model_diffs"] = KTCommon::CreateBlobWithSameValue(dim_vec,
//  4);
//  (*bn2blob_ptr)["model_expected"] =
//      KTCommon::CreateBlobWithSameValue(dim_vec, 3);
//  (*bn2blob_ptr)["momentum_expected"] =
//      KTCommon::CreateBlobWithSameValue(dim_vec, 1);
//  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
//}
//
// template<DeviceType device_type, typename FloatingPointType>
// void TestMomentumMdUpdateKernel() {
//  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
//  KernelCtx ctx;
//  KTCommon::BuildKernelCtx(&ctx);
//
//  const float learning_rate = {0.5f};
//  const float beta = {0.5f};
//  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, FloatingPointType>();
//  auto momentum_md_update_kernel =
//      BuildMomentumMdUpdateKernel<device_type,
//      FloatingPointType>(learning_rate,
//                                                                  beta);
//  int32_t piece_size = 1;
//  int32_t num_of_pieces_in_batch = 2;
//  InitJobDesc(piece_size, num_of_pieces_in_batch);
//
//  momentum_md_update_kernel->Forward(ctx, BnInOp2BlobPtr);
//  KTCommon::SyncStream(&ctx);
//
//  KTCommon::CheckResult(BnInOp2BlobPtr, "momentum", "momentum_expected");
//  KTCommon::CheckResult(BnInOp2BlobPtr, "model", "model_expected");
//}
//
//}  // namespace
//
//}  // namespace test
//
// TEST(MomentumMdUpdateKernel, model_update_cpu) {
//  test::TestMomentumMdUpdateKernel<DeviceType::kCPU, float>();
//  test::TestMomentumMdUpdateKernel<DeviceType::kCPU, double>();
//}
//
// TEST(MomentumMdUpdateKernel, model_update_gpu) {
//  test::TestMomentumMdUpdateKernel<DeviceType::kGPU, float>();
//  test::TestMomentumMdUpdateKernel<DeviceType::kGPU, double>();
//}
//
//}  // namespace oneflow
