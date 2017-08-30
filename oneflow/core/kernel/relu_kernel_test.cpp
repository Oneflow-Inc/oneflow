#include "oneflow/core/kernel/relu_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

template<typename T>
std::map<std::string, std::vector<T>> GenerateBlobMat();

template<>
std::map<std::string, std::vector<int32_t>> GenerateBlobMat<int32_t>() {
  std::map<std::string, std::vector<int32_t>> ret;
  std::vector<int32_t> in_mat = {1, -1, -2, 2, 0, 5, -10, 100};
  std::vector<int32_t> out_diff_mat = {-8, 7, -6, 5, -4, 3, -2, 1};
  std::vector<int32_t> out_mat(8, 0);
  std::vector<int32_t> in_diff_mat(8, 0);
  std::vector<int32_t> expected_out_mat = {1, 0, 0, 2, 0, 5, 0, 100};
  std::vector<int32_t> expected_in_diff_mat = {-8, 0, 0, 5, 0, 3, 0, 1};
  ret["in"] = in_mat;
  ret["out"] = out_mat;
  ret["in_diff"] = in_diff_mat;
  ret["out_diff"] = out_diff_mat;
  ret["expected_out"] = expected_out_mat;
  ret["expected_in_diff"] = expected_in_diff_mat;
  return ret;
}

template<>
std::map<std::string, std::vector<float>> GenerateBlobMat<float>() {
  std::map<std::string, std::vector<float>> ret;
  std::vector<float> in_mat = {1, -1, -2, 2, 0, 0.5, -10, 100};
  std::vector<float> out_diff_mat = {-8, 7, -6, 5, -4, 3, -2, 1};
  std::vector<float> out_mat(8, 0);
  std::vector<float> in_diff_mat(8, 0);
  std::vector<float> expected_out_mat = {1, 0, 0, 2, 0, 0.5, 0, 100};
  std::vector<float> expected_in_diff_mat = {-8, 0, 0, 5, 0, 3, 0, 1};
  ret["in"] = in_mat;
  ret["out"] = out_mat;
  ret["in_diff"] = in_diff_mat;
  ret["out_diff"] = out_diff_mat;
  ret["expected_out"] = expected_out_mat;
  ret["expected_in_diff"] = expected_in_diff_mat;
  return ret;
}

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
  DataType data_type = GetDataType<T>::val;
  BlobDesc* blob_desc = new BlobDesc(Shape({1, 8}), data_type, true);
  using KTC = KTCommon<device_type, T>;
  std::string data_id = "123456";
  JobConf tmp_job_conf;
  tmp_job_conf.set_max_data_id_length(12);
  JobDesc::Singleton()->InitFromJobConf(tmp_job_conf);
  auto blob_mat = GenerateBlobMat<T>();
  auto in_blob = KTC::CreateBlobWithSpecifiedVal(blob_desc, &blob_mat["in"][0]);
  auto out_diff_blob =
      KTC::CreateBlobWithSpecifiedVal(blob_desc, &blob_mat["out_diff"][0]);
  auto out_blob =
      KTC::CreateBlobWithSpecifiedVal(blob_desc, &blob_mat["out"][0]);
  auto in_diff_blob =
      KTC::CreateBlobWithSpecifiedVal(blob_desc, &blob_mat["in_diff"][0]);
  auto expected_out_blob =
      KTC::CreateBlobWithSpecifiedVal(blob_desc, &blob_mat["expected_out"][0]);
  auto expected_in_diff_blob = KTC::CreateBlobWithSpecifiedVal(
      blob_desc, &blob_mat["expected_in_diff"][0]);
  in_blob->template SetDataId<device_type>(0, data_id);
  out_blob->template SetDataId<device_type>(0, data_id);
  in_diff_blob->template SetDataId<device_type>(0, data_id);
  out_diff_blob->template SetDataId<device_type>(0, data_id);
  expected_out_blob->template SetDataId<device_type>(0, data_id);
  expected_in_diff_blob->template SetDataId<device_type>(0, data_id);
  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
  (*bn2blob_ptr)["in"] = in_blob;
  (*bn2blob_ptr)["out"] = out_blob;
  (*bn2blob_ptr)["in_diff"] = in_diff_blob;
  (*bn2blob_ptr)["out_diff"] = out_diff_blob;
  (*bn2blob_ptr)["expected_out"] = expected_out_blob;
  (*bn2blob_ptr)["expected_in_diff"] = expected_in_diff_blob;
  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename T>
Kernel* BuildReluKernel() {
  DataType data_type = GetDataType<T>::val;
  OperatorConf op_conf;
  op_conf.set_name("relu_op_test");
  ReluOpConf* relu_conf = op_conf.mutable_relu_conf();
  relu_conf->mutable_in()->set_name("relu/in");
  relu_conf->mutable_in()->set_data_type(data_type);
  relu_conf->mutable_out()->set_name("relu/out");
  relu_conf->mutable_out()->set_data_type(data_type);
  auto relu_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  relu_op->ToProto(&op_proto);
  auto relu_kernel = new ReluKernel<device_type, T>();
  relu_kernel->InitFromOpProto(op_proto);
  return relu_kernel;
}

template<DeviceType device_type, typename T>
void TestReluKernel() {
  using KTC = KTCommon<device_type, T>;
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);
  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, T>();
  auto relu_kernel = BuildReluKernel<device_type, T>();
  relu_kernel->Forward(ctx, BnInOp2BlobPtr);
  relu_kernel->Backward(ctx, BnInOp2BlobPtr);
  SyncStream<device_type>(&ctx);
  KTC::CheckResult(BnInOp2BlobPtr, "out", "expected_out");
  KTC::CheckResult(BnInOp2BlobPtr, "in_diff", "expected_in_diff");
}

}  // namespace test

TEST(ReluKernel, relu_kernel_cpu) {
  test::TestReluKernel<DeviceType::kCPU, float>();
  test::TestReluKernel<DeviceType::kCPU, int32_t>();
}

TEST(ReluKernel, relu_kernel_gpu) {
  test::TestReluKernel<DeviceType::kGPU, float>();
  test::TestReluKernel<DeviceType::kGPU, int32_t>();
}

}  // namespace oneflow
