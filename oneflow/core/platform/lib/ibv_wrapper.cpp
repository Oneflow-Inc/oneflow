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
#if defined(WITH_RDMA)
#include "oneflow/core/platform/include/ibv.h"

namespace oneflow {

namespace ibv {

std::vector<std::string> GetLibPaths() {
  const char* custom_path = std::getenv("ONEFLOW_LIBIBVERBS_PATH");
  if (custom_path == nullptr) {
    return {"libibverbs.so.1", "libibverbs.so"};
  } else {
    return {custom_path};
  }
}

platform::DynamicLibrary* GetIBVLibraryPtr() {
  static std::unique_ptr<platform::DynamicLibrary> lib =
      platform::DynamicLibrary::Load(GetLibPaths());
  return lib.get();
}

platform::DynamicLibrary& GetIBVLibrary() {
  platform::DynamicLibrary* lib = GetIBVLibraryPtr();
  CHECK(lib != nullptr) << "fail to find libibverbs";
  return *lib;
}

template<typename FUNC>
FUNC LoadSymbol(const char* name, FUNC* save) {
  auto fn = reinterpret_cast<FUNC>(GetIBVLibrary().LoadSym(name));
  if (!fn) {
    std::cerr << "Can't load libibverbs symbol " << name << "\n";
    abort();
  };
  *save = fn;
  return fn;
}

bool IsAvailable() { return GetIBVLibraryPtr() != nullptr; }

namespace _stubs {

void ibv_free_device_list(struct ibv_device** list) {
  return LoadSymbol(__func__, &wrapper.ibv_free_device_list)(list);
}

struct ibv_mr* ibv_reg_mr_wrap(struct ibv_pd* pd, void* addr, size_t length, int access) {
  return LoadSymbol("ibv_reg_mr", &wrapper.ibv_reg_mr_wrap)(pd, addr, length, access);
}

int ibv_destroy_qp(struct ibv_qp* qp) { return LoadSymbol(__func__, &wrapper.ibv_destroy_qp)(qp); }

int ibv_query_gid(struct ibv_context* context, uint8_t port_num, int index, union ibv_gid* gid) {
  return LoadSymbol(__func__, &wrapper.ibv_query_gid)(context, port_num, index, gid);
}

int ibv_fork_init(void) { return LoadSymbol(__func__, &wrapper.ibv_fork_init)(); }

int ibv_query_port_wrap(struct ibv_context* context, uint8_t port_num,
                        struct ibv_port_attr* port_attr) {
  return LoadSymbol("ibv_query_port", &wrapper.ibv_query_port_wrap)(context, port_num, port_attr);
}

struct ibv_context* ibv_open_device(struct ibv_device* device) {
  return LoadSymbol(__func__, &wrapper.ibv_open_device)(device);
}

int ibv_destroy_cq(struct ibv_cq* cq) { return LoadSymbol(__func__, &wrapper.ibv_destroy_cq)(cq); }

struct ibv_pd* ibv_alloc_pd(struct ibv_context* context) {
  return LoadSymbol(__func__, &wrapper.ibv_alloc_pd)(context);
}

int ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask) {
  return LoadSymbol(__func__, &wrapper.ibv_modify_qp)(qp, attr, attr_mask);
}

int ibv_dealloc_pd(struct ibv_pd* pd) { return LoadSymbol(__func__, &wrapper.ibv_dealloc_pd)(pd); }

struct ibv_device** ibv_get_device_list(int* num_devices) {
  return LoadSymbol(__func__, &wrapper.ibv_get_device_list)(num_devices);
}

int ibv_close_device(struct ibv_context* context) {
  return LoadSymbol(__func__, &wrapper.ibv_close_device)(context);
}

struct ibv_qp* ibv_create_qp(struct ibv_pd* pd, struct ibv_qp_init_attr* qp_init_attr) {
  return LoadSymbol(__func__, &wrapper.ibv_create_qp)(pd, qp_init_attr);
}

int ibv_dereg_mr(struct ibv_mr* mr) { return LoadSymbol(__func__, &wrapper.ibv_dereg_mr)(mr); }

struct ibv_cq* ibv_create_cq(struct ibv_context* context, int cqe, void* cq_context,
                             struct ibv_comp_channel* channel, int comp_vector) {
  return LoadSymbol(__func__, &wrapper.ibv_create_cq)(context, cqe, cq_context, channel,
                                                      comp_vector);
}

int ibv_query_device(struct ibv_context* context, struct ibv_device_attr* device_attr) {
  return LoadSymbol(__func__, &wrapper.ibv_query_device)(context, device_attr);
}

const char* ibv_get_device_name(struct ibv_device* device) {
  return LoadSymbol(__func__, &wrapper.ibv_get_device_name)(device);
}

}  // namespace _stubs

IBV wrapper = {
#define _REFERENCE_MEMBER(name) _stubs::name,
    IBV_APIS(_REFERENCE_MEMBER)
#undef _REFERENCE_MEMBER
        _stubs::ibv_reg_mr_wrap,
    _stubs::ibv_query_port_wrap};

}  // namespace ibv
}  // namespace oneflow
#endif  // WITH_RDMA
