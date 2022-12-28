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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template<typename L, typename P>
double RocAucScore(size_t n, const L* label, const P* pred, float* buffer) {
  size_t p_samples_count = 0;
  for (size_t i = 0; i < n; ++i) {
    if (label[i] == 0) {
      buffer[i] = -pred[i];
    } else {
      p_samples_count += 1;
      buffer[i] = pred[i];
    }
  }
  const size_t n_samples_count = n - p_samples_count;
  constexpr size_t kParallelSortThreshold = 1024;
  auto comp = [](float a, float b) { return fabs(a) < fabs(b); };
  if (n < kParallelSortThreshold) {
    std::sort(buffer, buffer + n, comp);
  } else {
    const size_t m2 = n / 2;
    const size_t m1 = m2 / 2;
    const size_t m3 = (m2 + n) / 2;
    std::thread t0([&] { std::sort(buffer, buffer + m1, comp); });
    std::thread t1([&] { std::sort(buffer + m1, buffer + m2, comp); });
    std::thread t2([&] { std::sort(buffer + m2, buffer + m3, comp); });
    std::thread t3([&] { std::sort(buffer + m3, buffer + n, comp); });
    t0.join();
    t1.join();
    t2.join();
    t3.join();
    std::inplace_merge(buffer, buffer + m1, buffer + m2, comp);
    std::inplace_merge(buffer + m2, buffer + m3, buffer + n, comp);
    std::inplace_merge(buffer, buffer + m2, buffer + n, comp);
  }
  size_t tmp_n = 0;
  double tmp_rank_sum = 0;
  double rank_sum = 0;
  size_t tmp_p_samples_count = 0;
  for (size_t i = 0; i < n; ++i) {
    if (i != 0 && fabs(buffer[i]) != fabs(buffer[i - 1])) {
      rank_sum += tmp_p_samples_count * (tmp_rank_sum / tmp_n);
      tmp_n = 0;
      tmp_rank_sum = 0;
      tmp_p_samples_count = 0;
    }
    if (buffer[i] > 0) { tmp_p_samples_count += 1; }
    tmp_rank_sum += (i + 1);
    tmp_n += 1;
  }
  rank_sum += tmp_p_samples_count * (tmp_rank_sum / tmp_n);
  return (rank_sum - p_samples_count * (p_samples_count + 1) / 2)
         / (p_samples_count * n_samples_count);
}

template<typename L, typename P>
class RocAucScoreKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RocAucScoreKernel);
  RocAucScoreKernel() = default;
  ~RocAucScoreKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    const user_op::Tensor* pred = ctx->Tensor4ArgNameAndIndex("pred", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    P* out_ptr = out->mut_dptr<P>();
    CHECK_EQ(label->shape_view().elem_cnt(), pred->shape_view().elem_cnt());
    CHECK_EQ(out->shape_view().elem_cnt(), 1);
    out_ptr[0] = RocAucScore(label->shape_view().elem_cnt(), label->dptr<L>(), pred->dptr<P>(),
                             tmp_buffer->mut_dptr<float>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ROC_AUC_SCORE_KERNEL(label_type, label_cpp_type, pred_type, pred_cpp_type) \
  REGISTER_USER_KERNEL("roc_auc_score")                                                     \
      .SetCreateFn<RocAucScoreKernel<label_cpp_type, pred_cpp_type>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                       \
                       && (user_op::HobDataType("label", 0) == label_type)                  \
                       && (user_op::HobDataType("pred", 0) == pred_type))                   \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                         \
        const Shape& pred_shape = ctx->InputShape("pred", 0);                               \
        size_t tmp_buffer_size = pred_shape.elem_cnt() * sizeof(float);                     \
        return tmp_buffer_size;                                                             \
      })
REGISTER_ROC_AUC_SCORE_KERNEL(DataType::kDouble, double, DataType::kFloat, float);
REGISTER_ROC_AUC_SCORE_KERNEL(DataType::kFloat, float, DataType::kFloat, float);
REGISTER_ROC_AUC_SCORE_KERNEL(DataType::kInt32, int, DataType::kFloat, float);
REGISTER_ROC_AUC_SCORE_KERNEL(DataType::kInt64, int64_t, DataType::kFloat, float);
REGISTER_ROC_AUC_SCORE_KERNEL(DataType::kInt8, int8_t, DataType::kFloat, float);
REGISTER_ROC_AUC_SCORE_KERNEL(DataType::kUInt8, uint8_t, DataType::kFloat, float);

}  // namespace

}  // namespace oneflow
