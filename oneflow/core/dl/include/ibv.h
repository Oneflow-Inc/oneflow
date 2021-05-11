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
#ifndef ONEFLOW_CORE_DL_INCLUDE_IBV_H_
#define ONEFLOW_CORE_DL_INCLUDE_IBV_H_
#include "oneflow/core/dl/include/wrapper.h"
#include <infiniband/verbs.h>

namespace oneflow {

namespace ibv {
// has to add extern otherwise it fails to compile at changes meaning of functions
extern "C" typedef struct IBV {
#define IBV_APIS(_)       \
  _(ibv_free_device_list) \
  _(ibv_destroy_qp)       \
  _(ibv_query_gid)        \
  _(ibv_fork_init)        \
  _(ibv_open_device)      \
  _(ibv_destroy_cq)       \
  _(ibv_alloc_pd)         \
  _(ibv_modify_qp)        \
  _(ibv_dealloc_pd)       \
  _(ibv_get_device_list)  \
  _(ibv_close_device)     \
  _(ibv_create_qp)        \
  _(ibv_dereg_mr)         \
  _(ibv_create_cq)        \
  _(ibv_query_device)     \
  _(ibv_reg_mr_iova2)

#define DECLARE_ONE(name) decltype(&name) name;
  IBV_APIS(DECLARE_ONE)
#undef DECLARE_ONE
  // for a function is not only a function but also a macro,
  // it requires an alternative name
  struct ibv_mr* (*ibv_reg_mr_)(struct ibv_pd* pd, void* addr, size_t length, int access);
  int (*ibv_query_port_)(struct ibv_context* context, uint8_t port_num,
                         struct _compat_ibv_port_attr* port_attr);
} IBV;

extern IBV wrapper;

// copy from infiniband/verbs.h
static inline int ___ibv_query_port(struct ibv_context* context, uint8_t port_num,
                                    struct ibv_port_attr* port_attr) {
  struct verbs_context* vctx = verbs_get_ctx_op(context, query_port);

  if (!vctx) {
    int rc;

    memset(port_attr, 0, sizeof(*port_attr));

    rc = wrapper.ibv_query_port_(context, port_num, (struct _compat_ibv_port_attr*)port_attr);
    return rc;
  }

  return vctx->query_port(context, port_num, port_attr, sizeof(*port_attr));
}

#undef ibv_query_port
#define ibv_query_port(context, port_num, port_attr) ___ibv_query_port(context, port_num, port_attr)

__attribute__((__always_inline__)) static inline struct ibv_mr* __ibv_reg_mr(
    struct ibv_pd* pd, void* addr, size_t length, unsigned int access, int is_access_const) {
  if (is_access_const && (access & IBV_ACCESS_OPTIONAL_RANGE) == 0)
    return wrapper.ibv_reg_mr_(pd, addr, length, access);
  else
    return wrapper.ibv_reg_mr_iova2(pd, addr, length, (uintptr_t)addr, access);
}

#undef ibv_reg_mr
#define ibv_reg_mr(pd, addr, length, access) \
  __ibv_reg_mr(pd, addr, length, access,     \
               __builtin_constant_p(((access)&IBV_ACCESS_OPTIONAL_RANGE) == 0))

}  // namespace ibv
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DL_INCLUDE_IBV_H_
#endif  // WITH_RDMA
