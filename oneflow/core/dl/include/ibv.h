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
  _(ibv_reg_mr)           \
  _(ibv_destroy_qp)       \
  _(ibv_query_gid)        \
  _(ibv_fork_init)        \
  _(ibv_query_port)       \
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
} NVRTC;

extern IBV wrapper;

}  // namespace ibv
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DL_INCLUDE_IBV_H_
#endif  // WITH_RDMA
