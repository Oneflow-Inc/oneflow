#include "oneflow/core/kernel/normal_model_update_kernel.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename FloatingPointType>
Kernel* BuildMdUpdateKernel(float learning_rate) {
  OperatorConf op_conf;
  op_conf.set_name("model_update_test");
  NormalModelUpdateOpConf* model_update_conf =
      op_conf.mutable_normal_mdupdt_conf();
  model_update_conf->set_learning_rate(learning_rate);
  auto model_update_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  model_update_op->ToProto(&op_proto);
  auto model_update_kernel =
      new MdUpdateKernel<device_type, FloatingPointType>();
  model_update_kernel->InitFromOpProto(op_proto);
  return model_update_kernel;
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

  std::vector<int64_t> dim_vec = {1, 3, 2};

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
  (*bn2blob_ptr)["model"] = KTCommon::CreateBlobWithSameValue(dim_vec, 2);
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

  const float learning_rate = {1.0f};
  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, FloatingPointType>();
  auto model_update_kernel =
      BuildMdUpdateKernel<device_type, FloatingPointType>(learning_rate);
  int32_t piece_size = 1;
  int32_t num_of_pieces_in_batch = 2;
  InitJobDesc(piece_size, num_of_pieces_in_batch);

  model_update_kernel->Forward(ctx, BnInOp2BlobPtr);
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
