#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type>
void RngUniform(std::vector<float>* random_mask, int64_t seed) {
  TODO();
}

template<DeviceType device_type, typename T>
std::vector<T> ExpectedDropout(const std::vector<T>& input, float rate,
                               int64_t seed) {
  std::vector<T> output(input);
  std::vector<float> random_mask(input.size());
  RngUniform<device_type>(&random_mask, seed);
  FOR_RANGE(int, i, 0, output.size()) {
    output.at(i) *= random_mask.at(i) < rate;
  }
  return output;
}

}  // namespace

template<DeviceType device_type, typename T>
OpKernelTestCase* DropoutTestCase(const std::string& job_type,
                                  const std::string& fw_or_bw) {
  OpKernelTestCase* dropout_test_case = new OpKernelTestCase();
  dropout_test_case->set_is_train(job_type == "train");
  dropout_test_case->set_is_forward(forward_or_backward == "forward");
  dropout_test_case->set_device_type(device_type);
  auto* dropout_conf = dropout_test_case->mut_op_conf()->mutable_droput_conf();
  float rate = 0.5;
  dropout_conf->set_rate(rate);
  float seed = 0xdeadbeef;
  dropout_conf->set_seed(seed);

  using KTC = KTCommon<device_type, T>;
  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 8}), GetDataType<T>::val, false, false, 1);
  std::vector<T> input_data{1, -1, -2, 2, 0, 5, -10, 100};
  std::vector<T> output_diff_data{-8, 7, -6, 5, -4, 3, -2, 1};
  dropout_test_case->InitBlob(
      "in", KTC::CreateBlobWithSpecifiedVal(blob_desc, input_data));
  dropout_test_case->InitBlob(
      GenDiffBn("out"),
      KTC::CreateBlobWithSpecifiedVal(blob_desc, output_diff_data));
  dropout_test_case->ForwardCheckBlob(
      "out", device_type,
      KTC::CreateBlobWithSpecifiedVal(
          blob_desc, ExpectedDropout<device_type, T>(input_data, rate, seed)));
  dropout_test_case->BackwardCheckBlob(
      GenDiffBn("in"), device_type,
      KTC::CreateBlobWithSpecifiedVal(
          blob_desc,
          ExpectedDropout<device_type, T>(output_diff_data, rate, seed)));

  return nullptr;
}

TEST_CPU_ONLY_OPKERNEL(DropoutTestCase, ARITHMETIC_DATA_TYPE_SEQ,
                       (train)(predict), (forward)(backward));

}  // namespace test

}  // namespace oneflow
