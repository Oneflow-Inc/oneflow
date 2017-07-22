#include "oneflow/core/kernel/salr_model_update_kernel.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename FloatingPointType>
Kernel* BuildSALRMdUpdateKernel(float delta, float epsilon) {
  OperatorConf op_conf;
  op_conf.set_name("salr_model_update_test");
  SALRModelUpdateOpConf* salr_md_update_conf =
      op_conf.mutable_salr_mdupdt_conf();
  salr_md_update_conf->set_delta(delta);
  salr_md_update_conf->set_epsilon(epsilon);
  auto salr_md_update_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  salr_md_update_op->ToProto(&op_proto);
  auto salr_md_update_kernel =
      new SALRMdUpdateKernel<device_type, FloatingPointType>();
  salr_md_update_kernel->InitFromOpProto(op_proto);
  return salr_md_update_kernel;
}

void InitJobDesc(int32_t piece_size, int32_t num_of_pieces_in_batch) {
  JobConf job_conf;
  job_conf.set_piece_size(piece_size);
  job_conf.set_num_of_pieces_in_batch(num_of_pieces_in_batch);
  JobDesc::Singleton()->InitFromJobConf(job_conf);
}

template<DeviceType device_type, typename FloatingPointType>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;

  std::vector<int64_t> dim_vec = {1, 2, 2};

  FloatingPointType model[] = {4, 4, 4, 4};
  FloatingPointType model_expected[] = {-2, 2, 6, 10};
  FloatingPointType last_diff_flag[] = {1, -1, 1, -1};
  FloatingPointType last_diff_flag_expected[] = {1, 1, -1, -1};
  FloatingPointType model_diff[] = {4, 4, -4, -4};
  FloatingPointType learning_rate[] = {1, 1, 1, 1};
  FloatingPointType learning_rate_expected[] = {1.5, 0.5, 0.5, 1.5};

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
  (*bn2blob_ptr)["model"] = KTCommon::CreateBlobWithVector(dim_vec, model);
  (*bn2blob_ptr)["model_expected"] =
      KTCommon::CreateBlobWithVector(dim_vec, model_expected);
  (*bn2blob_ptr)["last_diff_flag"] =
      KTCommon::CreateBlobWithVector(dim_vec, last_diff_flag);
  (*bn2blob_ptr)["last_diff_flag_expected"] =
      KTCommon::CreateBlobWithVector(dim_vec, last_diff_flag_expected);
  (*bn2blob_ptr)["model_diff"] =
      KTCommon::CreateBlobWithVector(dim_vec, model_diff);
  (*bn2blob_ptr)["learning_rate"] =
      KTCommon::CreateBlobWithVector(dim_vec, learning_rate);
  (*bn2blob_ptr)["learning_rate_expected"] =
      KTCommon::CreateBlobWithVector(dim_vec, learning_rate_expected);
  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename FloatingPointType>
void TestSALRMdUpdateKernel() {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);

  const float delta = {0.5f};
  const float epsilon = {-2.0f};

  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, FloatingPointType>();
  auto salr_md_update_kernel =
      BuildSALRMdUpdateKernel<device_type, FloatingPointType>(delta, epsilon);

  int32_t piece_size = 1;
  int32_t num_of_pieces_in_batch = 2;
  InitJobDesc(piece_size, num_of_pieces_in_batch);

  salr_md_update_kernel->Forward(ctx, BnInOp2BlobPtr);
  KTCommon::SyncStream(&ctx);

  KTCommon::CheckResult(BnInOp2BlobPtr, "learning_rate",
                        "learning_rate_expected");
  KTCommon::CheckResult(BnInOp2BlobPtr, "model", "model_expected");
  KTCommon::CheckResult(BnInOp2BlobPtr, "last_diff_flag",
                        "last_diff_flag_expected");
}

}  // namespace

}  // namespace test

TEST(SALRMdUpdateKernel, model_update_cpu) {
  test::TestSALRMdUpdateKernel<DeviceType::kCPU, float>();
  test::TestSALRMdUpdateKernel<DeviceType::kCPU, double>();
}

TEST(SALRMdUpdateKernel, model_update_gpu) {
  test::TestSALRMdUpdateKernel<DeviceType::kGPU, float>();
  test::TestSALRMdUpdateKernel<DeviceType::kGPU, double>();
}

}  // namespace oneflow
