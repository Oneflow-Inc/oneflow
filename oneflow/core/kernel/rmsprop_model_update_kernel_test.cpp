//#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"
//#include "oneflow/core/kernel/kernel_test_common.h"
//
// namespace oneflow {
//
// namespace test {
//
// namespace {
//
// template<DeviceType device_type, typename FloatingPointType>
// Kernel* BuildRMSPropMdUpdateKernel(float learning_rate, float decay_rate,
//                                   float epsilon) {
//  OperatorConf op_conf;
//  op_conf.set_name("rmsprop_model_update_test");
//  RMSPropModelUpdateOpConf* rmsprop_md_update_conf =
//      op_conf.mutable_rmsprop_mdupdt_conf();
//  rmsprop_md_update_conf->set_learning_rate(learning_rate);
//  rmsprop_md_update_conf->set_decay_rate(decay_rate);
//  rmsprop_md_update_conf->set_epsilon(epsilon);
//  auto rmsprop_md_update_op = ConstructOp(op_conf);
//  OperatorProto op_proto;
//  rmsprop_md_update_op->ToProto(&op_proto);
//  auto rmsprop_md_update_kernel =
//      new RMSPropMdUpdateKernel<device_type, FloatingPointType>();
//  rmsprop_md_update_kernel->InitFromOpProto(op_proto);
//  return rmsprop_md_update_kernel;
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
// std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr(
//    std::vector<int64_t>& dim_vec) {
//  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
//
//  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
//  (*bn2blob_ptr)["model"] = KTCommon::CreateBlobWithSameValue(dim_vec, 2);
//  (*bn2blob_ptr)["mean_square"] = KTCommon::CreateBlobWithSameValue(dim_vec,
//  0);
//  (*bn2blob_ptr)["model_diffs"] = KTCommon::CreateBlobWithSameValue(dim_vec,
//  2);
//  (*bn2blob_ptr)["model_expected"] =
//      KTCommon::CreateBlobWithSameValue(dim_vec, 0);
//  (*bn2blob_ptr)["mean_square_expected"] =
//      KTCommon::CreateBlobWithSameValue(dim_vec, 1);
//  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
//}
//
// template<DeviceType device_type, typename FloatingPointType>
// void TestRMSPropMdUpdateKernel() {
//  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
//  KernelCtx ctx;
//  KTCommon::BuildKernelCtx(&ctx);
//  ctx.other = new int64_t(1);
//
//  std::vector<int64_t> dim_vec = {1, 3, 2};
//  const float learning_rate = {2.0f};
//  const float decay_rate = 1.0f / 2;
//  const float epsilon = 3.0f;
//  auto BnInOp2BlobPtr =
//      BuildBnInOp2BlobPtr<device_type, FloatingPointType>(dim_vec);
//  auto rmsprop_md_update_kernel =
//      BuildRMSPropMdUpdateKernel<device_type, FloatingPointType>(
//          learning_rate, decay_rate, epsilon);
//  int32_t piece_size = 1;
//  int32_t num_of_pieces_in_batch = 2;
//  InitJobDesc(piece_size, num_of_pieces_in_batch);
//
//  rmsprop_md_update_kernel->Forward(ctx, BnInOp2BlobPtr);
//  ctx.other = new int64_t(2);
//  rmsprop_md_update_kernel->Forward(ctx, BnInOp2BlobPtr);
//  KTCommon::SyncStream(&ctx);
//
//  KTCommon::CheckResult(BnInOp2BlobPtr, "mean_square",
//  "mean_square_expected"); KTCommon::CheckResult(BnInOp2BlobPtr, "model",
//  "model_expected");
//}
//
//}  // namespace
//
//}  // namespace test
//
// TEST(RMSPropMdUpdateKernel, model_update_cpu) {
//  test::TestRMSPropMdUpdateKernel<DeviceType::kCPU, float>();
//  test::TestRMSPropMdUpdateKernel<DeviceType::kCPU, double>();
//}
//
// TEST(RMSPropMdUpdateKernel, model_update_gpu) {
//  test::TestRMSPropMdUpdateKernel<DeviceType::kGPU, float>();
//  test::TestRMSPropMdUpdateKernel<DeviceType::kGPU, double>();
//}
//
//}  // namespace oneflow
