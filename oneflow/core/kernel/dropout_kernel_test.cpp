#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/kernel/random_generator.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type>
void RngUniform(std::vector<float>* random_mask, int64_t seed);

template<>
void RngUniform<DeviceType::kCPU>(std::vector<float>* random_mask,
                                  int64_t seed) {
  RandomGenerator random_generator(seed);
  random_generator.Uniform<DeviceType::kCPU, float>(random_mask->size(),
                                                    &random_mask->front());
}

template<>
void RngUniform<DeviceType::kGPU>(std::vector<float>* random_mask,
                                  int64_t seed) {
  float* mem_ptr = nullptr;
  size_t buf_size = random_mask->size() * sizeof(float);
  CudaCheck(cudaMalloc(&mem_ptr, buf_size));
  RandomGenerator random_generator(seed);
  random_generator.Uniform<DeviceType::kGPU, float>(random_mask->size(),
                                                    mem_ptr);
  CudaCheck(cudaMemcpy(&random_mask->front(), mem_ptr, buf_size,
                       cudaMemcpyDeviceToHost));
}

template<DeviceType device_type, typename T>
std::vector<T> Dropout(const std::vector<T>& input,
                       const std::vector<float>& random_mask, float rate) {
  std::vector<T> output(input);
  FOR_RANGE(int, i, 0, output.size()) {
    output.at(i) *= (random_mask.at(i) > rate);
  }
  return output;
}

}  // namespace

template<DeviceType device_type, typename T>
OpKernelTestCase* DropoutTestCase(const std::string& job_type,
                                  const std::string& fw_or_bw) {
  OpKernelTestCase* dropout_test_case = new OpKernelTestCase();
  dropout_test_case->set_is_train(job_type == "train");
  dropout_test_case->set_is_forward(fw_or_bw == "forward");
  dropout_test_case->set_device_type(device_type);
  auto* dropout_conf = dropout_test_case->mut_op_conf()->mutable_dropout_conf();
  float rate = 0.5;
  dropout_conf->set_rate(rate);
  float seed = 0xdeadbeef;
  dropout_conf->set_seed(seed);

  using KTC = KTCommon<device_type, T>;
  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 8}), GetDataType<T>::val, false, false, 1);
  std::vector<T> input_data{1, -1, -2, 2, 0, 5, -10, 100};
  std::vector<T> output_diff_data{-8, 7, -6, 5, -4, 3, -2, 1};
  std::vector<float> random_mask(input_data.size());
  RngUniform<device_type>(&random_mask, seed);
  dropout_test_case->InitBlob(
      "in", KTC::CreateBlobWithSpecifiedVal(blob_desc, input_data));
  dropout_test_case->InitBlob(
      GenDiffBn("out"),
      KTC::CreateBlobWithSpecifiedVal(blob_desc, output_diff_data));
  dropout_test_case->ForwardCheckBlob(
      "out", device_type,
      KTC::CreateBlobWithSpecifiedVal(
          blob_desc, Dropout<device_type, T>(input_data, random_mask, rate)));
  dropout_test_case->BackwardCheckBlob(
      GenDiffBn("in"), device_type,
      KTC::CreateBlobWithSpecifiedVal(
          blob_desc,
          Dropout<device_type, T>(output_diff_data, random_mask, rate)));
  return dropout_test_case;
}

TEST_CPU_ONLY_OPKERNEL(DropoutTestCase, ARITHMETIC_DATA_TYPE_SEQ,
                       (train)(predict), (forward)(backward));

}  // namespace test

}  // namespace oneflow
