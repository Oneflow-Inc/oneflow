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
#if defined(WITH_CUDA) && defined(USE_HIP)

#include <gtest/gtest.h>
#include <cudnn.h>
#include <vector>

namespace oneflow {

namespace test {

namespace {

constexpr int kN = 2, kC = 3, kH = 9, kW = 9, kK = 4, kR = 3, kS = 3;

class ConvProblem {
 public:
  ConvProblem() {
    EXPECT_EQ(cudnnCreateTensorDescriptor(&x_desc_), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnCreateTensorDescriptor(&y_desc_), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnCreateFilterDescriptor(&w_desc_), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnCreateConvolutionDescriptor(&conv_desc_), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(
        cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, kN, kC, kH, kW),
        CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(
        cudnnSetTensor4dDescriptor(y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, kN, kK, kH, kW),
        CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(
        cudnnSetFilter4dDescriptor(w_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kK, kC, kR, kS),
        CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnSetConvolution2dDescriptor(conv_desc_, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT),
              CUDNN_STATUS_SUCCESS);
    x_.resize(kN * kC * kH * kW);
    w_.resize(kK * kC * kR * kS);
    for (size_t i = 0; i < x_.size(); ++i) { x_[i] = static_cast<float>(i % 7) - 3.f; }
    for (size_t i = 0; i < w_.size(); ++i) { w_[i] = static_cast<float>(i % 5) - 2.f; }
    EXPECT_EQ(hipMalloc(&x_dev_, x_.size() * sizeof(float)), hipSuccess);
    EXPECT_EQ(hipMalloc(&w_dev_, w_.size() * sizeof(float)), hipSuccess);
    EXPECT_EQ(hipMalloc(&y_dev_, kN * kK * kH * kW * sizeof(float)), hipSuccess);
    EXPECT_EQ(hipMemcpy(x_dev_, x_.data(), x_.size() * sizeof(float), hipMemcpyHostToDevice),
              hipSuccess);
    EXPECT_EQ(hipMemcpy(w_dev_, w_.data(), w_.size() * sizeof(float), hipMemcpyHostToDevice),
              hipSuccess);
  }

  ~ConvProblem() {
    (void)hipFree(x_dev_);
    (void)hipFree(w_dev_);
    (void)hipFree(y_dev_);
    (void)cudnnDestroyConvolutionDescriptor(conv_desc_);
    (void)cudnnDestroyFilterDescriptor(w_desc_);
    (void)cudnnDestroyTensorDescriptor(y_desc_);
    (void)cudnnDestroyTensorDescriptor(x_desc_);
  }

  // Exact in float: every product and partial sum is a small integer.
  std::vector<float> Reference() const {
    std::vector<float> y(kN * kK * kH * kW, 0.f);
    for (int n = 0; n < kN; ++n) {
      for (int k = 0; k < kK; ++k) {
        for (int h = 0; h < kH; ++h) {
          for (int w = 0; w < kW; ++w) {
            float sum = 0.f;
            for (int c = 0; c < kC; ++c) {
              for (int r = 0; r < kR; ++r) {
                for (int s = 0; s < kS; ++s) {
                  const int ih = h + r - 1;
                  const int iw = w + s - 1;
                  if (ih < 0 || ih >= kH || iw < 0 || iw >= kW) { continue; }
                  sum +=
                      x_[((n * kC + c) * kH + ih) * kW + iw] * w_[((k * kC + c) * kR + r) * kS + s];
                }
              }
            }
            y[((n * kK + k) * kH + h) * kW + w] = sum;
          }
        }
      }
    }
    return y;
  }

  cudnnTensorDescriptor_t x_desc_ = nullptr;
  cudnnTensorDescriptor_t y_desc_ = nullptr;
  cudnnFilterDescriptor_t w_desc_ = nullptr;
  cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
  std::vector<float> x_;
  std::vector<float> w_;
  void* x_dev_ = nullptr;
  void* w_dev_ = nullptr;
  void* y_dev_ = nullptr;
};

}  // namespace

// oneflow picks the algorithm on one handle and runs on the stream's own handle. A convolution on
// a handle created after another was destroyed must still run, whatever address it gets. On an
// MI250X with ROCm 10.1, miopenCreate returned an address it had returned before about once in 60
// create/destroy cycles, so 256 rounds reach that case.
TEST(HipCudnn, ConvolutionOnRecreatedHandles) {
  ConvProblem p;
  cudnnHandle_t search_handle = nullptr;
  ASSERT_EQ(cudnnCreate(&search_handle), CUDNN_STATUS_SUCCESS);
  cudnnConvolutionFwdAlgoPerf_t perf[8];
  int count = 0;
  ASSERT_EQ(cudnnGetConvolutionForwardAlgorithm_v7(search_handle, p.x_desc_, p.w_desc_,
                                                   p.conv_desc_, p.y_desc_, 8, &count, perf),
            CUDNN_STATUS_SUCCESS);
  ASSERT_GT(count, 0);
  size_t workspace_size = 0;
  ASSERT_EQ(
      cudnnGetConvolutionForwardWorkspaceSize(search_handle, p.x_desc_, p.w_desc_, p.conv_desc_,
                                              p.y_desc_, perf[0].algo, &workspace_size),
      CUDNN_STATUS_SUCCESS);
  void* workspace = nullptr;
  ASSERT_EQ(hipMalloc(&workspace, workspace_size == 0 ? 1 : workspace_size), hipSuccess);
  const std::vector<float> reference = p.Reference();
  const float one = 1.f;
  const float zero = 0.f;
  for (int round = 0; round < 256; ++round) {
    cudnnHandle_t handle = nullptr;
    ASSERT_EQ(cudnnCreate(&handle), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipMemset(p.y_dev_, 0, reference.size() * sizeof(float)), hipSuccess);
    ASSERT_EQ(cudnnConvolutionForward(handle, &one, p.x_desc_, p.x_dev_, p.w_desc_, p.w_dev_,
                                      p.conv_desc_, perf[0].algo, workspace, workspace_size, &zero,
                                      p.y_desc_, p.y_dev_),
              CUDNN_STATUS_SUCCESS)
        << "round " << round;
    std::vector<float> y(reference.size());
    ASSERT_EQ(hipMemcpy(y.data(), p.y_dev_, y.size() * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(y, reference) << "round " << round;
    ASSERT_EQ(cudnnDestroy(handle), CUDNN_STATUS_SUCCESS);
  }
  (void)hipFree(workspace);
  ASSERT_EQ(cudnnDestroy(search_handle), CUDNN_STATUS_SUCCESS);
}

// Whether miopenCreate reuses an address is up to the allocator, so the record itself is checked
// too: once a handle is destroyed, the same address must count as a handle with no Find run.
TEST(HipCudnn, FindRecordDroppedWithHandle) {
  ConvProblem p;
  cudnnHandle_t handle = nullptr;
  ASSERT_EQ(cudnnCreate(&handle), CUDNN_STATUS_SUCCESS);
  EXPECT_TRUE(oneflow_hip_cudnn::FirstRunOnHandle(handle, 0, p.x_desc_, p.w_desc_->desc, p.y_desc_,
                                                  p.conv_desc_->desc));
  EXPECT_FALSE(oneflow_hip_cudnn::FirstRunOnHandle(handle, 0, p.x_desc_, p.w_desc_->desc, p.y_desc_,
                                                   p.conv_desc_->desc));
  ASSERT_EQ(cudnnDestroy(handle), CUDNN_STATUS_SUCCESS);
  // Only the address is used here; the destroyed handle is not passed to MIOpen.
  EXPECT_TRUE(oneflow_hip_cudnn::FirstRunOnHandle(handle, 0, p.x_desc_, p.w_desc_->desc, p.y_desc_,
                                                  p.conv_desc_->desc));
  oneflow_hip_cudnn::FindRecords* records = oneflow_hip_cudnn::GetFindRecords();
  std::lock_guard<std::mutex> lock(records->mutex);
  records->problems.erase(handle);
}

}  // namespace test

}  // namespace oneflow

#endif  // defined(WITH_CUDA) && defined(USE_HIP)
