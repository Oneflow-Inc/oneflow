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
#ifndef ONEFLOW_CORE_MEMORY_MEMORY_CASE_REGISTRY_H_
#define ONEFLOW_CORE_MEMORY_MEMORY_CASE_REGISTRY_H_

#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/framework/device_registry.h"

namespace oneflow {

template<typename registry_t>
class MemCaseRegistryMgr {
 public:
  registry_t& Register(DeviceType device_type) {
    auto ret = items_.emplace(device_type, registry_t{});
    CHECK(ret.second);
    return ret.first->second;
  }

  const registry_t& LookupRegistry(typename registry_t::key_t key) const {
    const registry_t* registry_ptr = nullptr;
    for (const auto& pair : items_) {
      if (pair.second.Match(key)) {
        CHECK(registry_ptr == nullptr) << "found multiple matching registry";
        registry_ptr = &pair.second;
      }
    }
    CHECK(registry_ptr != nullptr) << "not found matching registry";
    return *registry_ptr;
  }

  static MemCaseRegistryMgr& Get() {
    static MemCaseRegistryMgr mgr;
    return mgr;
  }

 private:
  HashMap<DeviceType, registry_t> items_;
};

template<DeviceType device_type, typename registry_t>
struct MemCaseRegisterHelper {
  MemCaseRegisterHelper(registry_t registry) {
    MemCaseRegistryMgr<registry_t>::Get().Register(device_type) = std::move(registry);
  }
};

class MemCaseIdGeneratorRegistry final {
 public:
  using self_t = MemCaseIdGeneratorRegistry;
  using key_t = const MemoryCase&;
  using matcher_t = std::function<bool(key_t)>;
  using generator_t = std::function<MemCaseId(key_t)>;

  self_t& SetMatcher(matcher_t matcher) {
    matcher_ = std::move(matcher);
    return *this;
  }
  bool Match(key_t key) const { return matcher_(key); }

  self_t& SetGenerator(generator_t generator) {
    generator_ = std::move(generator);
    return *this;
  }
  generator_t::result_type Generate(key_t key) const { return generator_(key); }

 private:
  matcher_t matcher_;
  generator_t generator_;
};

class MemCaseIdToProtoRegistry final {
 public:
  using self_t = MemCaseIdToProtoRegistry;
  using key_t = const MemCaseId&;
  using matcher_t = std::function<bool(key_t)>;
  using to_proto_t = std::function<void(key_t, MemoryCase*)>;

  self_t& SetMatcher(matcher_t matcher) {
    matcher_ = std::move(matcher);
    return *this;
  }
  bool Match(key_t key) const { return matcher_(key); }

  self_t& SetToProto(to_proto_t to_proto) {
    to_proto_ = std::move(to_proto);
    return *this;
  }
  void ToProto(key_t key, MemoryCase* mem_case) const { to_proto_(key, mem_case); }

 private:
  matcher_t matcher_;
  to_proto_t to_proto_;
};

class PageLockedMemCaseRegistry final {
 public:
  using self_t = PageLockedMemCaseRegistry;
  using key_t = const MemoryCase&;
  using matcher_t = std::function<bool(key_t)>;
  using page_locker_t = std::function<void(key_t, MemoryCase*)>;

  self_t& SetMatcher(matcher_t matcher) {
    matcher_ = std::move(matcher);
    return *this;
  }
  bool Match(key_t key) const { return matcher_(key); }

  self_t& SetPageLocker(page_locker_t page_locker) {
    page_locker_ = std::move(page_locker);
    return *this;
  }
  void PageLock(key_t key, MemoryCase* page_locked_mem_case) const {
    return page_locker_(key, page_locked_mem_case);
  }

 private:
  matcher_t matcher_;
  page_locker_t page_locker_;
};

class PatchMemCaseRegistry final {
 public:
  using self_t = PatchMemCaseRegistry;
  using key_t = const MemoryCase&;
  using matcher_t = std::function<bool(key_t)>;
  using patcher_t = std::function<bool(key_t, MemoryCase*)>;

  self_t& SetMatcher(matcher_t matcher) {
    matcher_ = std::move(matcher);
    return *this;
  }
  bool Match(key_t key) const { return matcher_(key); }

  self_t& SetPatcher(patcher_t patcher) {
    patcher_ = std::move(patcher);
    return *this;
  }
  bool Patch(key_t key, MemoryCase* dst_mem_case) const { return patcher_(key, dst_mem_case); }

 private:
  matcher_t matcher_;
  patcher_t patcher_;
};

}  // namespace oneflow

#define REGISTER_MEM_CASE_ID_GENERATOR(device_type_v)                                           \
  static ::oneflow::MemCaseRegisterHelper<device_type_v, ::oneflow::MemCaseIdGeneratorRegistry> \
  OF_PP_CAT(g_mem_case_register_helper, __COUNTER__) = ::oneflow::MemCaseIdGeneratorRegistry {}

#define REGISTER_MEM_CASE_ID_TO_PROTO(device_type_v)                                          \
  static ::oneflow::MemCaseRegisterHelper<device_type_v, ::oneflow::MemCaseIdToProtoRegistry> \
  OF_PP_CAT(g_mem_case_register_helper, __COUNTER__) = ::oneflow::MemCaseIdToProtoRegistry {}

#define REGISTER_PAGE_LOCKED_MEM_CASE(device_type_v)                                           \
  static ::oneflow::MemCaseRegisterHelper<device_type_v, ::oneflow::PageLockedMemCaseRegistry> \
  OF_PP_CAT(g_mem_case_register_helper, __COUNTER__) = ::oneflow::PageLockedMemCaseRegistry {}

#define REGISTER_PATCH_MEM_CASE(device_type_v)                                            \
  static ::oneflow::MemCaseRegisterHelper<device_type_v, ::oneflow::PatchMemCaseRegistry> \
  OF_PP_CAT(g_mem_case_register_helper, __COUNTER__) = ::oneflow::PatchMemCaseRegistry {}

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_CASE_REGISTRY_H_
