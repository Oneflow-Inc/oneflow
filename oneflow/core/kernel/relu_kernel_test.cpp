#include "oneflow/core/kernel/relu_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
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
    blob_desc->set_has_data_id(false);
    using KTCommon = KTCommon<device_type, float>;
    float in_mat[8] = {1.0, -1.0, -2.0, 2.0, 0.0, 0.5, -10.0, 100.0};
    float out_diff_mat[8] = {-8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0};
    float out_mat[8] = {0};
    float in_diff_mat[8] = {0};
    float expected_out_mat[8] = {1.0, 0, 0, 2.0, 0, 0.5, 0, 100.0};
    float expected_in_diff_mat[8] = {-8.0, 0, 0, 5.0, 0, 3.0, 0, 1.0};
    auto bn2blob_ptr = new HashMap<std::string, Blob*>;
    (*bn2blob_ptr)["in"] =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, in_mat);
    (*bn2blob_ptr)["out"] =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, out_mat);
    (*bn2blob_ptr)["in_diff"] =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, in_diff_mat);
    (*bn2blob_ptr)["out_diff"] =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, out_diff_mat);
    (*bn2blob_ptr)["expected_out"] =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, expected_out_mat);
    (*bn2blob_ptr)["expected_in_diff"] =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, expected_in_diff_mat);
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
    blob_desc->set_has_data_id(false);
    using KTCommon = KTCommon<device_type, int32_t>;
    int32_t in_mat[8] = {1, -1, -2, 2, 0, 5, -10, 100};
    int32_t out_diff_mat[8] = {-8, 7, -6, 5, -4, 3, -2, 1};
    int32_t out_mat[8] = {0};
    int32_t in_diff_mat[8] = {0};
    int32_t expected_out_mat[8] = {1, 0, 0, 2, 0, 5, 0, 100};
    int32_t expected_in_diff_mat[8] = {-8, 0, 0, 5, 0, 3, 0, 1};
    auto bn2blob_ptr = new HashMap<std::string, Blob*>;
    (*bn2blob_ptr)["in"] =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, in_mat);
    (*bn2blob_ptr)["out"] =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, out_mat);
    (*bn2blob_ptr)["in_diff"] =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, in_diff_mat);
    (*bn2blob_ptr)["out_diff"] =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, out_diff_mat);
    (*bn2blob_ptr)["expected_out"] =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, expected_out_mat);
    (*bn2blob_ptr)["expected_in_diff"] =
        KTCommon::CreateBlobWithSpecifiedVal(blob_desc, expected_in_diff_mat);
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
