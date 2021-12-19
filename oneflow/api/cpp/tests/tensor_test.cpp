/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "oneflow/api/cpp/tests/api_test.h"
#include <gtest/gtest.h>

namespace oneflow_api {

TEST(Api, device) {
  EnvScope scope;

  auto device = Device("cpu");
  ASSERT_EQ(device.type(), "cpu");

#ifdef WITH_CUDA
  device = Device("cuda:0");
  ASSERT_EQ(device.type(), "cuda");
  ASSERT_EQ(device.device_id(), 0);

  device = Device("cuda", 1);
  ASSERT_EQ(device.type(), "cuda");
  ASSERT_EQ(device.device_id(), 1);
#endif
}

TEST(Api, tensor) {
  EnvScope scope;

  const auto device = Device("cpu");
  const auto shape = RandomShape();
  const auto dtype = DType::kDouble;

  Tensor tensor;
  ASSERT_EQ(tensor.shape(), Shape());
  ASSERT_EQ(tensor.device(), Device("cpu"));
  ASSERT_EQ(tensor.dtype(), DType::kFloat);

  Tensor tensor_with_all(shape, device, dtype);

  ASSERT_EQ(tensor_with_all.shape(), shape);
  ASSERT_EQ(tensor_with_all.device(), device);
  ASSERT_EQ(tensor_with_all.dtype(), dtype);
}

TEST(Api, tensor_from_buffer_and_copy_to) {
  EnvScope scope;

  const auto shape = RandomShape();

#define TEST_TENSOR_FROM_AND_TO_BLOB(dtype, cpp_dtype)                                           \
  std::vector<cpp_dtype> data_##cpp_dtype(shape.Count(0)), new_data_##cpp_dtype(shape.Count(0)); \
  for (int i = 0; i < shape.Count(0); ++i) { data_##cpp_dtype[i] = i; }                          \
  auto tensor_##cpp_dtype =                                                                      \
      Tensor::from_buffer(data_##cpp_dtype.data(), shape, Device("cpu"), dtype);                 \
  tensor_##cpp_dtype.copy_to(new_data_##cpp_dtype.data());                                       \
  ASSERT_EQ(new_data_##cpp_dtype, data_##cpp_dtype);

  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kFloat, float)
  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kDouble, double)
  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kInt8, int8_t)
  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kInt32, int32_t)
  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kInt64, int64_t)
}

TEST(Api, tensor_zeros) {
  EnvScope scope;

  const auto shape = RandomShape();

  std::vector<float> data(shape.Count(0)), target_data(shape.Count(0));

  Tensor tensor(shape, Device("cpu"), DType::kFloat);
  tensor.zeros_();
  tensor.copy_to(data.data());

  std::fill(target_data.begin(), target_data.end(), 0);

  ASSERT_EQ(data, target_data);
}
void TestTensorPrint(bool on_gpu) {
#define TEST_TENSOR_PRINT(dtype, cpp_dtype)                                        \
  const auto shape_##cpp_dtype = RandomSmallShape();                               \
  const auto data_##cpp_dtype = RandomData<cpp_dtype>(shape_##cpp_dtype.Count(0)); \
  const auto tensor_##cpp_dtype =                                                  \
      Tensor::from_buffer(data_##cpp_dtype.data(), shape_##cpp_dtype,              \
                          on_gpu ? Device("cuda:0") : Device("cpu"), dtype);       \
  std::cout << tensor_##cpp_dtype << std::endl;

  TEST_TENSOR_PRINT(DType::kFloat, float)
  TEST_TENSOR_PRINT(DType::kDouble, double)
  TEST_TENSOR_PRINT(DType::kInt8, int8_t)
  TEST_TENSOR_PRINT(DType::kInt32, int32_t)
  TEST_TENSOR_PRINT(DType::kInt64, int64_t)
}

TEST(Api, tensor_print) {
  EnvScope scope;
  TestTensorPrint(/*one_gpu*/ false);

#ifdef WITH_CUDA
  TestTensorPrint(/*on_gpu*/ true);
#endif
}

void TestTensorPrintConfirmValue(bool on_gpu) {
  const auto get_tensor_printed_string = [](const Tensor& tensor) -> std::string {
    std::string temp_str, result_str;
    std::stringstream ss;
    ss << tensor;
    while (std::getline(ss, temp_str, '\n')) {
      if (result_str != "") { result_str += "\n"; }
      result_str += temp_str;
    }
    return result_str;
  };

  const auto build_target_str = [](const std::string& str, bool on_gpu,
                                   const std::string& data_type) -> std::string {
    return str + (on_gpu ? "cuda:0" : "cpu") + ", DataType: " + data_type + "]";
  };

  const auto device = on_gpu ? Device("cuda:0") : Device("cpu");
  const std::vector<float> data_float{1.1,  2.2,   -3.3,  4.44,     5.55,    6.66,
                                      7.77, 8.888, 9.999, -10.0000, 11.1111, 12.1212};
  const std::vector<int32_t> data_int32{8, 100, 55, 88, 9, -5, 2, -7};

  const auto tensor_float = Tensor::from_buffer(data_float.data(), {3, 4}, device, DType::kFloat);
  const auto tensor_int32 =
      Tensor::from_buffer(data_int32.data(), {2, 2, 2}, device, DType::kInt32);

  const auto tensor_float_str = build_target_str(R"(  1.1000   2.2000  -3.3000   4.4400
  5.5500   6.6600   7.7700   8.8880
  9.9990 -10.0000  11.1111  12.1212
[Shape: (3,4), Device: )",
                                                 on_gpu, "Float");

  const auto tensor_int32_str = build_target_str(R"((1,.,.) = 
    8  100
   55   88

(2,.,.) = 
  9 -5
  2 -7
[Shape: (2,2,2), Device: )",
                                                 on_gpu, "Int32");

  const auto result_str_float = get_tensor_printed_string(tensor_float);
  const auto result_str_int32 = get_tensor_printed_string(tensor_int32);

  ASSERT_EQ(result_str_float, tensor_float_str);
  ASSERT_EQ(result_str_int32, tensor_int32_str);
}

TEST(Api, tensor_print_confirm_value) {
  EnvScope scope;
  TestTensorPrintConfirmValue(/*one_gpu*/ false);

#ifdef WITH_CUDA
  TestTensorPrintConfirmValue(/*on_gpu*/ true);
#endif
}

}  // namespace oneflow_api
