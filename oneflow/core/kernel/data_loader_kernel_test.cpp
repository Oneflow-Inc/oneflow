//#include "oneflow/core/kernel/data_loader_kernel.h"
//#include "oneflow/core/common/process_state.h"
//#include "oneflow/core/common/str_util.h"
//#include "oneflow/core/job/runtime_context.h"
//#include "oneflow/core/kernel/kernel_test_common.h"
//
// namespace oneflow {
//
// namespace test {
//
// namespace {
//
// template<typename FloatingPointType>
// Kernel* BuildDataLoaderKernel() {
//  OperatorConf op_conf;
//  op_conf.set_name("data_loader_test");
//  DataLoaderOpConf* data_loader_conf = op_conf.mutable_data_loader_conf();
//  data_loader_conf->set_data_dir("");
//  auto data_loader_op = ConstructOp(op_conf);
//  OperatorProto op_proto;
//  data_loader_op->ToProto(&op_proto);
//  auto data_loader_kernel =
//      new DataLoaderKernel<DeviceType::kCPU, FloatingPointType>();
//  data_loader_kernel->InitFromOpProto(op_proto);
//  return data_loader_kernel;
//}
//
// template<typename FloatingPointType>
// std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
//  using KTCommon = KernelTestCommon<DeviceType::kCPU, FloatingPointType>;
//
//  std::vector<int64_t> dim_vec = {1, 3, 2};
//
//  std::vector<FloatingPointType> label_expected = {1, 2, 4, 7, 11, 12};
//  std::vector<FloatingPointType> feature_expected = {3, 5, 6, 8, 9, 10};
//
//  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
//  (*bn2blob_ptr)["label"] = KTCommon::CreateBlobWithSameValue(dim_vec, 0);
//  (*bn2blob_ptr)["feature"] = KTCommon::CreateBlobWithSameValue(dim_vec, 0);
//  (*bn2blob_ptr)["label_expected"] =
//      KTCommon::CreateBlobWithVector(dim_vec, &label_expected.front());
//  (*bn2blob_ptr)["feature_expected"] =
//      KTCommon::CreateBlobWithVector(dim_vec, &feature_expected.front());
//
//  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
//}
//
// void InitFile(const std::string& filepath) {
//  std::unique_ptr<tensorflow::WritableFile> file;
//  TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(filepath, &file));
//  file->Append("1\n");
//  file->Append("2,3\n");
//  file->Append("4,5,6\n");
//  file->Append("7,8,9,10\n");
//  file->Append("11\n");
//  file->Append("12");
//  TF_CHECK_OK(file->Close());
//}
//
// template<typename FloatingPointType>
// void TestDataLoaderKernel() {
//  using KTCommon = KernelTestCommon<DeviceType::kCPU, FloatingPointType>;
//  KernelCtx ctx;
//  KTCommon::BuildKernelCtx(&ctx);
//
//  std::string current_dir = GetCwd();
//  StringReplace(&current_dir, '\\', '/');
//  std::string data_loader_root_dir =
//      JoinPath(current_dir, "/data_loader_test_tmp_dir");
//  TF_CHECK_OK(tensorflow::Env::Default()->CreateDir(data_loader_root_dir));
//  std::string filepath = JoinPath(current_dir, "/tmp_file");
//  InitFile(filepath);
//  RuntimeCtx::Singleton()->InitDataReader(filepath);
//
//  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<FloatingPointType>();
//  auto data_loader_kernel = BuildDataLoaderKernel<FloatingPointType>();
//
//  data_loader_kernel->Forward(ctx, BnInOp2BlobPtr);
//  KTCommon::SyncStream(&ctx);
//
//  KTCommon::CheckResult(BnInOp2BlobPtr, "label", "label_expected");
//  KTCommon::CheckResult(BnInOp2BlobPtr, "feature", "feature_expected");
//
//  tensorflow::int64 undeletefiles, undeletedirs;
//  TF_CHECK_OK(tensorflow::Env::Default()->DeleteRecursively(
//      data_loader_root_dir, &undeletefiles, &undeletedirs));
//}
//
//}  // namespace
//
//}  // namespace test
//
// TEST(DataLoaderKernel, data_loader_cpu) {
//  test::TestDataLoaderKernel<float>();
//  test::TestDataLoaderKernel<double>();
//}
//
//}  // namespace oneflow
