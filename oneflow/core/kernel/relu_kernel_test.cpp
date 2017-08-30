#include "oneflow/core/kernel/relu_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
class ReluTestUtil final {
 public:
  static std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr();
};

template<DeviceType device_type>
class ReluTestUtil<device_type, float> final {
 public:
  static std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
    DataType data_type = DataType::kFloat;
    BlobDesc* blob_desc = new BlobDesc();
    blob_desc->mut_shape() = Shape({1, 8});
    blob_desc->set_data_type(data_type);
    blob_desc->set_has_data_id(true);
    using KTCommon = KTCommon<device_type, float>;
    JobConf tmp_job_conf;
    tmp_job_conf.set_max_data_id_length(12);
    JobDesc::Singleton()->InitFromJobConf(tmp_job_conf);
    std::string data_id = "123456";
    float in_mat[8] = {1.0, -1.0, -2.0, 2.0, 0.0, 0.5, -10.0, 100.0};
    float out_diff_mat[8] = {-8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0};
    float out_mat[8] = {0};
    float in_diff_mat[8] = {0};
    float expected_out_mat[8] = {1.0, 0, 0, 2.0, 0, 0.5, 0, 100.0};
    float expected_in_diff_mat[8] = {-8.0, 0, 0, 5.0, 0, 3.0, 0, 1.0};
    auto in_blob = KTCommon::CreateBlobWithSpecifiedVal(blob_desc, in_mat);
    auto out_diff_blob =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, out_diff_mat);
    auto out_blob = KTCommon::CreateBlobWithSpecifiedVal(blob_desc, out_mat);
    auto in_diff_blob =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, in_diff_mat);
    auto expected_out_blob =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, expected_out_mat);
    auto expected_in_diff_blob =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, expected_in_diff_mat);
    in_blob->set_data_id(0, data_id);
    out_blob->set_data_id(0, data_id);
    in_diff_blob->set_data_id(0, data_id);
    out_diff_blob->set_data_id(0, data_id);
    expected_out_blob->set_data_id(0, data_id);
    expected_in_diff_blob->set_data_id(0, data_id);
    auto bn2blob_ptr = new HashMap<std::string, Blob*>;
    (*bn2blob_ptr)["in"] = in_blob;
    (*bn2blob_ptr)["out"] = out_blob;
    (*bn2blob_ptr)["in_diff"] = in_diff_blob;
    (*bn2blob_ptr)["out_diff"] = out_diff_blob;
    (*bn2blob_ptr)["expected_out"] = expected_out_blob;
    (*bn2blob_ptr)["expected_in_diff"] = expected_in_diff_blob;
    return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
  }
};

template<DeviceType device_type>
class ReluTestUtil<device_type, int32_t> final {
 public:
  static std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
    DataType data_type = DataType::kInt32;
    BlobDesc* blob_desc = new BlobDesc();
    blob_desc->mut_shape() = Shape({1, 8});
    blob_desc->set_data_type(data_type);
    blob_desc->set_has_data_id(true);
    using KTCommon = KTCommon<device_type, int32_t>;
    std::string data_id = "123456";
    JobConf tmp_job_conf;
    tmp_job_conf.set_max_data_id_length(12);
    JobDesc::Singleton()->InitFromJobConf(tmp_job_conf);
    int32_t in_mat[8] = {1, -1, -2, 2, 0, 5, -10, 100};
    int32_t out_diff_mat[8] = {-8, 7, -6, 5, -4, 3, -2, 1};
    int32_t out_mat[8] = {0};
    int32_t in_diff_mat[8] = {0};
    int32_t expected_out_mat[8] = {1, 0, 0, 2, 0, 5, 0, 100};
    int32_t expected_in_diff_mat[8] = {-8, 0, 0, 5, 0, 3, 0, 1};
    auto in_blob = KTCommon::CreateBlobWithSpecifiedVal(blob_desc, in_mat);
    auto out_diff_blob =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, out_diff_mat);
    auto out_blob = KTCommon::CreateBlobWithSpecifiedVal(blob_desc, out_mat);
    auto in_diff_blob =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, in_diff_mat);
    auto expected_out_blob =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, expected_out_mat);
    auto expected_in_diff_blob =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, expected_in_diff_mat);
    in_blob->set_data_id(0, data_id);
    out_blob->set_data_id(0, data_id);
    in_diff_blob->set_data_id(0, data_id);
    out_diff_blob->set_data_id(0, data_id);
    expected_out_blob->set_data_id(0, data_id);
    expected_in_diff_blob->set_data_id(0, data_id);
    auto bn2blob_ptr = new HashMap<std::string, Blob*>;
    (*bn2blob_ptr)["in"] = in_blob;
    (*bn2blob_ptr)["out"] = out_blob;
    (*bn2blob_ptr)["in_diff"] = in_diff_blob;
    (*bn2blob_ptr)["out_diff"] = out_diff_blob;
    (*bn2blob_ptr)["expected_out"] = expected_out_blob;
    (*bn2blob_ptr)["expected_in_diff"] = expected_in_diff_blob;
    return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
  }
};

template<DeviceType device_type, typename T>
Kernel* BuildReluKernel() {
  DataType data_type = GetDataType<T>::val;
  OperatorConf op_conf;
  op_conf.set_name("relu_op_test");
  ReluOpConf* relu_conf = op_conf.mutable_relu_conf();
  LogicalBlob* in_blob = new LogicalBlob();
  in_blob->set_name("relu/in");
  in_blob->set_data_type(data_type);
  LogicalBlob* out_blob = new LogicalBlob();
  out_blob->set_name("relu/out");
  out_blob->set_data_type(data_type);
  *(relu_conf->mutable_in()) = *in_blob;
  *(relu_conf->mutable_out()) = *out_blob;
  auto relu_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  relu_op->ToProto(&op_proto);
  auto relu_kernel = new ReluKernel<device_type, T>();
  relu_kernel->InitFromOpProto(op_proto);
  return relu_kernel;
}

template<DeviceType device_type, typename T>
void TestReluKernel() {
  using KTCommon = KTCommon<device_type, T>;
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);
  auto BnInOp2BlobPtr = ReluTestUtil<device_type, T>::BuildBnInOp2BlobPtr();
  auto relu_kernel = BuildReluKernel<device_type, T>();
  relu_kernel->Forward(ctx, BnInOp2BlobPtr);
  relu_kernel->Backward(ctx, BnInOp2BlobPtr);
  SyncStream<device_type>(&ctx);
  KTCommon::CheckResult(BnInOp2BlobPtr, "out", "expected_out");
  KTCommon::CheckResult(BnInOp2BlobPtr, "in_diff", "expected_in_diff");
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
