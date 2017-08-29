#include "oneflow/core/kernel/normal_model_update_kernel.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename T>
Kernel* BuildMdUpdateKernel(float learning_rate) {
  OperatorConf op_conf;
  op_conf.set_name("model_update_test");
  NormalModelUpdateOpConf* model_update_conf =
      op_conf.mutable_normal_mdupdt_conf();
  model_update_conf->set_learning_rate(learning_rate);
  auto model_update_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  model_update_op->ToProto(&op_proto);
  auto model_update_kernel = new NormalMdUpdateKernel<device_type, T>();
  model_update_kernel->InitFromOpProto(op_proto);
  return model_update_kernel;
}

void InitJobDesc(int32_t piece_size, int32_t num_of_pieces_in_batch) {
  JobConf job_conf;
  job_conf.set_piece_size(piece_size);
  job_conf.set_num_of_pieces_in_batch(num_of_pieces_in_batch);
  JobDesc::Singleton()->InitFromJobConf(job_conf);
}

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2Blob() {
  using KTCommon = KTCommon<device_type, T>;

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 3, 2}), GetDataType<T>::val, false);

  auto bn2blob = new HashMap<std::string, Blob*>;
  (*bn2blob)["model"] = KTCommon::CreateBlobWithSameVal(blob_desc, 2);
  (*bn2blob)["model_diffs"] = KTCommon::CreateBlobWithSameVal(blob_desc, 2);
  (*bn2blob)["model_expected"] = KTCommon::CreateBlobWithSameVal(blob_desc, 1);
  return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
}

template<DeviceType device_type, typename T>
void TestMdUpdateKernel() {
  using KTCommon = KTCommon<device_type, T>;
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);

  const float learning_rate = {1.0f};
  auto BnInOp2BlobPtr = BuildBnInOp2Blob<device_type, T>();
  auto model_update_kernel = BuildMdUpdateKernel<device_type, T>(learning_rate);
  int32_t piece_size = 1;
  int32_t num_of_pieces_in_batch = 2;
  InitJobDesc(piece_size, num_of_pieces_in_batch);

  model_update_kernel->Forward(ctx, BnInOp2BlobPtr);
  SyncStream<device_type>(&ctx);

  KTCommon::CheckResult(BnInOp2BlobPtr, "model", "model_expected");
}

}  // namespace

}  // namespace test

TEST(MdUpdateKernel, model_update) {
#define FLOATINGPOINTTYPE (double)(float)
#define DEVICETYPE (DeviceType::kCPU)(DeviceType::kGPU)
#define TESTMdUpdateKernel(x, y) test::TestMdUpdateKernel<x, y>();
  FOR_EACH_TUPLE(TESTMdUpdateKernel,
                 OF_PP_INTERNAL_SEQ_PRODUCT(DEVICETYPE, FLOATINGPOINTTYPE));
}

}  // namespace oneflow
