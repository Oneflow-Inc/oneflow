#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/kernel/random_generator.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type>
void RngUniform(std::vector<float>* random_mask, int64_t seed);

template<>
void RngUniform<DeviceType::kCPU>(std::vector<float>* random_mask, int64_t seed) {
  RandomGenerator random_generator(seed);
  random_generator.Uniform<DeviceType::kCPU, float>(random_mask->size(), &random_mask->front());
}

#ifdef WITH_CUDA
template<>
void RngUniform<DeviceType::kGPU>(std::vector<float>* random_mask, int64_t seed) {
  float* mem_ptr = nullptr;
  size_t buf_size = random_mask->size() * sizeof(float);
  CudaCheck(cudaMalloc(&mem_ptr, buf_size));
  RandomGenerator random_generator(seed);
  random_generator.Uniform<DeviceType::kGPU, float>(random_mask->size(), mem_ptr);
  CudaCheck(cudaMemcpy(&random_mask->front(), mem_ptr, buf_size, cudaMemcpyDeviceToHost));
}
#endif

template<DeviceType device_type, typename T>
std::vector<T> Dropout(const std::vector<T>& input, const std::vector<float>& random_mask,
                       float rate) {
  std::vector<T> output(input);
  FOR_RANGE(int, i, 0, output.size()) { output.at(i) *= (random_mask.at(i) > rate) / (1.f - rate); }
  return output;
}

}  // namespace

template<DeviceType device_type, typename T>
void DropoutTestCase(OpKernelTestCase* dropout_test_case, const std::string& job_type,
                     const std::string& fw_or_bw) {
  bool is_train = (job_type == "train");
  dropout_test_case->set_is_train(is_train);
  dropout_test_case->set_is_forward(fw_or_bw == "forward");
  auto* dropout_conf = dropout_test_case->mut_op_conf()->mutable_dropout_conf();
  float rate = 0.5;
  dropout_conf->set_rate(rate);
  float seed = 0xdeadbeef;
  dropout_conf->set_seed(seed);

  BlobDesc* blob_desc = new BlobDesc(Shape({1, 8}), GetDataType<T>::value, false, false, 1);
  BlobDesc* random_mask_blob_desc =
      new BlobDesc(Shape({1, 8}), GetDataType<float>::value, false, false, 1);
  std::vector<T> input_data{1, -1, -2, 2, 30, 5, -10, 100};
  std::vector<T> output_diff_data{-8, 7, -6, 5, -4, 3, -2, 1};
  std::vector<float> random_mask(input_data.size());
  RngUniform<device_type>(&random_mask, seed);
  dropout_test_case->template InitBlob<T>("in", blob_desc, input_data);
  dropout_test_case->template InitBlob<T>(GenDiffBn("out"), blob_desc, output_diff_data);
  if (is_train) {
    dropout_test_case->template ForwardCheckBlob<float>("random_mask", random_mask_blob_desc,
                                                        random_mask);
  }
  dropout_test_case->template ForwardCheckBlob<T>(
      "out", blob_desc, Dropout<device_type, T>(input_data, random_mask, (is_train ? rate : 0.f)));
  dropout_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), blob_desc,
      Dropout<device_type, T>(output_diff_data, random_mask, (is_train ? rate : 0.f)));
}

TEST_CPU_AND_GPU_OPKERNEL(DropoutTestCase, ARITHMETIC_DATA_TYPE_SEQ, (train)(predict),
                          (forward)(backward));

}  // namespace test

}  // namespace oneflow
