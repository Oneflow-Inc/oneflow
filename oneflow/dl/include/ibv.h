#ifndef ONEFLOW_DL_INCLUDE_IBV_H_
#define ONEFLOW_DL_INCLUDE_IBV_H_
#include "oneflow/dl/include/wrapper.h"

namespace oneflow {
namespace ibv {

struct IBV {
#define IBV_APIS(_) _(ibv_fork_init)

#define DECLARE_ONE(name) decltype(&name) name;
  IBV_APIS(DECLARE_ONE)
#undef DECLARE_ONE
};

extern IBV wrapper;

}  // namespace ibv
}  // namespace oneflow

#endif  // ONEFLOW_DL_INCLUDE_IBV_H_
