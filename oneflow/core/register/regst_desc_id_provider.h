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
#ifndef ONEFLOW_CORE_REGISTER_REGST_DESC_ID_PROVIDER_H_
#define ONEFLOW_CORE_REGISTER_REGST_DESC_ID_PROVIDER_H_

#include <atomic>
#include <memory>

namespace oneflow {

class RegstDescIdProvider {
 public:
  RegstDescIdProvider() = default;
  virtual ~RegstDescIdProvider() = default;

  virtual int64_t regst_desc_id() const = 0;
  virtual bool has_regst_desc_id() const = 0;
};

class ConstRegstDescIdProvider final : public RegstDescIdProvider {
 public:
  explicit ConstRegstDescIdProvider(int64_t regst_desc_id) : regst_desc_id_(regst_desc_id) {}
  ConstRegstDescIdProvider();
  ~ConstRegstDescIdProvider() override = default;

  int64_t regst_desc_id() const override { return regst_desc_id_; }

  bool has_regst_desc_id() const override { return true; }

 private:
  int64_t regst_desc_id_;
};

class LazyInitRegstDescIdProvider final : public RegstDescIdProvider {
 public:
  LazyInitRegstDescIdProvider() : RegstDescIdProvider(), regst_desc_id_(0) {}
  ~LazyInitRegstDescIdProvider() override = default;

  int64_t regst_desc_id() const override;

  bool has_regst_desc_id() const override {
    return regst_desc_id_.load(std::memory_order_acquire) != 0;
  }

  void init_regst_desc_id();
  void init_regst_desc_id(int64_t regst_desc_id);

 private:
  std::atomic<int64_t> regst_desc_id_;
};

std::unique_ptr<RegstDescIdProvider> NewRegstDescIdProvider();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_REGST_DESC_ID_PROVIDER_H_
