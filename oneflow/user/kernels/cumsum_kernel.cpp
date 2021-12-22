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
#include "unistd.h"
#include <thread>
#include <future>

namespace oneflow {

namespace {
template<typename T>
void cumsum_forward_norm(const T* pin, T* pout, int64_t cs_up_space, int64_t cs_space,
                         int64_t cs_down_space, int64_t elem_cnt) {
  std::copy_n(pin, elem_cnt, pout);
  for (auto i = 0; i < cs_up_space; i++) {
    auto* tmp_pout_base = pout + i * cs_space * cs_down_space;
    for (auto j = 1; j < cs_space; j++) {
      auto* tmp_pout = tmp_pout_base + j * cs_down_space;
      auto* last_tmp_pout = tmp_pout - cs_down_space;
      for (auto k = 0; k < cs_down_space; k++) { tmp_pout[k] += last_tmp_pout[k]; }
    }
  }
}

template<typename T>
void cumsum_forward_thread(const T* pin, T* pout, int64_t cs_up_space, int64_t cs_space,
                           int64_t cs_down_space, int64_t elem_cnt) {
  auto CPU_NUM = sysconf(_SC_NPROCESSORS_CONF);

  std::vector<std::future<void>> rets;
  rets.resize(CPU_NUM);
  auto njobs = cs_up_space * cs_down_space;
  auto njobs_per_thr = njobs / CPU_NUM;
  if (cs_up_space * cs_down_space % CPU_NUM) njobs_per_thr += 1;
  // start threads according to CPU number
  // total jobs number are cs_up_space * cs_down_space
  // every thread handls ceil(njobs / CPU_NUM) jobs
  for (auto thr_id = 0; thr_id < CPU_NUM; thr_id++) {
    rets[thr_id] = std::async(
        std::launch::async,
        [&](auto thr_id) {
          for (auto i = thr_id * njobs_per_thr; i < ((thr_id + 1) * njobs_per_thr) && i < njobs;
               i++) {
            auto cs_up_space_id = i / cs_down_space;
            auto cs_down_space_id = i % cs_down_space;

            auto* pin_base = pin + cs_up_space_id * cs_space * cs_down_space + cs_down_space_id;
            auto* pout_base = pout + cs_up_space_id * cs_space * cs_down_space + cs_down_space_id;

            // calculate cs_space data in one thread
            for (auto j = 0; j < cs_space; j++) {
              auto idx = j * cs_down_space;
              pout_base[idx] = pin_base[idx];
              if (j != 0) { pout_base[idx] += pout_base[idx - cs_down_space]; }
            }
          }
        },
        thr_id);
  }

  for (auto i = 0; i < CPU_NUM; i++) { rets[i].wait(); }
}

template<typename T>
void cumsum_backward_norm(const T* pin, T* pout, int64_t cs_up_space, int64_t cs_space,
                          int64_t cs_down_space, int64_t elem_cnt) {
  for (auto i = 0; i < cs_up_space; i++) {
    auto* tmp_pin_base = pin + i * cs_space * cs_down_space;
    auto* tmp_pout_base = pout + i * cs_space * cs_down_space;
    for (auto j = 0; j < cs_space; j++) {
      auto* tmp_pin = tmp_pin_base + j * cs_down_space;
      auto* tmp_pout = tmp_pout_base + j * cs_down_space;
      std::fill_n(tmp_pout, cs_down_space, cs_space - j);
      for (auto k = 0; k < cs_down_space; k++) { tmp_pout[k] *= tmp_pin[k]; }
    }
  }
}
}  // namespace

template<typename T>
class CpuCumsumKernel final : public user_op::OpKernel {
 public:
  CpuCumsumKernel() = default;
  ~CpuCumsumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // judge whether tensor has 0 size dimension first
    const auto* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    auto elem_cnt = in->shape().elem_cnt();
    if (!elem_cnt) { return; }

    auto* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto dim = ctx->Attr<int64_t>("dim");
    const auto* pin = in->dptr<T>();
    auto* pout = out->mut_dptr<T>();

    // take cumsum's abbreviation as `cs`
    // data partition: cs_up_space|cs_space|cs_down_space
    auto cs_up_space = elem_cnt / in->shape().Count(dim);
    auto cs_space = in->shape().At(dim);
    auto cs_down_space = in->shape().Count(dim) / cs_space;

    // cumsum_forward_norm<T>(pin, pout, cs_up_space, cs_space, cs_down_space, elem_cnt);
    cumsum_forward_thread<T>(pin, pout, cs_up_space, cs_space, cs_down_space, elem_cnt);
  }

  // TODO: what's it used for?
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUMSUM_KERNEL(dtype)                                                   \
  REGISTER_USER_KERNEL("cumsum").SetCreateFn<CpuCumsumKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                    \
      && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CUMSUM_KERNEL(float)
REGISTER_CUMSUM_KERNEL(double)

template<typename T>
class CpuCumsumGradKernel final : public user_op::OpKernel {
 public:
  CpuCumsumGradKernel() = default;
  ~CpuCumsumGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto elem_cnt = dy->shape().elem_cnt();
    auto dim = ctx->Attr<int64_t>("dim");
    const auto* dy_ptr = dy->dptr<T>();
    auto* dx_ptr = dx->mut_dptr<T>();

    // data partition: cs_up_space|cs_space|cs_down_space
    auto cs_up_space = elem_cnt / dx->shape().Count(dim);
    auto cs_space = dx->shape().At(dim);
    auto cs_down_space = dx->shape().Count(dim) / cs_space;

    cumsum_backward_norm(dy_ptr, dx_ptr, cs_up_space, cs_space, cs_down_space, elem_cnt);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_CUMSUM_GRAD_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("cumsum_grad")                                 \
      .SetCreateFn<CpuCumsumGradKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CPU_CUMSUM_GRAD_KERNEL(float)
REGISTER_CPU_CUMSUM_GRAD_KERNEL(double)

}  // namespace oneflow
