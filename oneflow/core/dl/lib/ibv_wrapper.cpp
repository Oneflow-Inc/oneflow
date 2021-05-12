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
#include "oneflow/core/dl/include/ibv.h"

namespace oneflow {

namespace dl {

DynamicLibrary& getIBVLibrary() {
  static std::unique_ptr<DynamicLibrary> lib =
      DynamicLibrary::Load({"libibverbs.so.1", "libibverbs.so"});
  CHECK(lib != nullptr) << "fail to find libibverbs";
  return *lib;
}

}  // namespace dl

namespace ibv {

namespace _stubs {

void ibv_free_device_list(struct ibv_device** list) {
  auto fn =
      reinterpret_cast<decltype(&ibv_free_device_list)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_free_device_list = fn;
  return fn(list);
}

struct ibv_mr* ibv_reg_mr_(struct ibv_pd* pd, void* addr, size_t length, int access) {
  auto fn = reinterpret_cast<struct ibv_mr* (*)(struct ibv_pd*, void*, size_t, int)>(
      dl::getIBVLibrary().LoadSym("ibv_reg_mr"));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_reg_mr_ = fn;
  return fn(pd, addr, length, access);
}

int ibv_destroy_qp(struct ibv_qp* qp) {
  auto fn = reinterpret_cast<decltype(&ibv_destroy_qp)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_destroy_qp = fn;
  return fn(qp);
}

int ibv_query_gid(struct ibv_context* context, uint8_t port_num, int index, union ibv_gid* gid) {
  auto fn = reinterpret_cast<decltype(&ibv_query_gid)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_query_gid = fn;
  return fn(context, port_num, index, gid);
}

int ibv_fork_init(void) {
  auto fn = reinterpret_cast<decltype(&ibv_fork_init)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_fork_init = fn;
  return fn();
}

int ibv_query_port_(struct ibv_context* context, uint8_t port_num,
                    struct ibv_port_attr* port_attr) {
  auto fn = reinterpret_cast<int (*)(struct ibv_context*, uint8_t, struct ibv_port_attr*)>(
      dl::getIBVLibrary().LoadSym("ibv_query_port"));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_query_port_ = fn;
  return fn(context, port_num, port_attr);
}

struct ibv_context* ibv_open_device(struct ibv_device* device) {
  auto fn = reinterpret_cast<decltype(&ibv_open_device)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_open_device = fn;
  return fn(device);
}

int ibv_destroy_cq(struct ibv_cq* cq) {
  auto fn = reinterpret_cast<decltype(&ibv_destroy_cq)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_destroy_cq = fn;
  return fn(cq);
}

struct ibv_pd* ibv_alloc_pd(struct ibv_context* context) {
  auto fn = reinterpret_cast<decltype(&ibv_alloc_pd)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_alloc_pd = fn;
  return fn(context);
}

int ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask) {
  auto fn = reinterpret_cast<decltype(&ibv_modify_qp)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_modify_qp = fn;
  return fn(qp, attr, attr_mask);
}

int ibv_dealloc_pd(struct ibv_pd* pd) {
  auto fn = reinterpret_cast<decltype(&ibv_dealloc_pd)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_dealloc_pd = fn;
  return fn(pd);
}

struct ibv_device** ibv_get_device_list(int* num_devices) {
  auto fn = reinterpret_cast<decltype(&ibv_get_device_list)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_get_device_list = fn;
  return fn(num_devices);
}

int ibv_close_device(struct ibv_context* context) {
  auto fn = reinterpret_cast<decltype(&ibv_close_device)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_close_device = fn;
  return fn(context);
}

struct ibv_qp* ibv_create_qp(struct ibv_pd* pd, struct ibv_qp_init_attr* qp_init_attr) {
  auto fn = reinterpret_cast<decltype(&ibv_create_qp)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_create_qp = fn;
  return fn(pd, qp_init_attr);
}

int ibv_dereg_mr(struct ibv_mr* mr) {
  auto fn = reinterpret_cast<decltype(&ibv_dereg_mr)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_dereg_mr = fn;
  return fn(mr);
}

struct ibv_cq* ibv_create_cq(struct ibv_context* context, int cqe, void* cq_context,
                             struct ibv_comp_channel* channel, int comp_vector) {
  auto fn = reinterpret_cast<decltype(&ibv_create_cq)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_create_cq = fn;
  return fn(context, cqe, cq_context, channel, comp_vector);
}

int ibv_query_device(struct ibv_context* context, struct ibv_device_attr* device_attr) {
  auto fn = reinterpret_cast<decltype(&ibv_query_device)>(dl::getIBVLibrary().LoadSym(__func__));
  if (!fn) { LOG(FATAL) << "Can't get ibv"; };
  wrapper.ibv_query_device = fn;
  return fn(context, device_attr);
}

}  // namespace _stubs

IBV wrapper = {
#define _REFERENCE_MEMBER(name) _stubs::name,
    IBV_APIS(_REFERENCE_MEMBER)
#undef _REFERENCE_MEMBER
        _stubs::ibv_reg_mr_,
    _stubs::ibv_query_port_};

}  // namespace ibv
}  // namespace oneflow
#endif  // WITH_RDMA
