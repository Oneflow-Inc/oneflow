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
  _(ibv_query_device)

#define DECLARE_ONE(name) decltype(&name) name;
  IBV_APIS(DECLARE_ONE)
#undef DECLARE_ONE
  // ibv_reg_mr or ibv_query_port is not only a function but also a macro
  // so we have to have alternative names
  decltype(&ibv_reg_mr) ibv_reg_mr_;
  int (*ibv_query_port_)(struct ibv_context* context, uint8_t port_num,
                         struct ibv_port_attr* port_attr);
} IBV;

extern IBV wrapper;

}  // namespace ibv
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DL_INCLUDE_IBV_H_
#endif  // WITH_RDMA
