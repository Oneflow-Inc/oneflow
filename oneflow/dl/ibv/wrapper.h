#if defined(WITH_RDMA)

#include <infiniband/verbs.h>

namespace oneflow {

struct IBV {
#define IBV_APIS(_) \
  _(ibv_free_dm)    \
  _(ibv_memcpy_to_dm)

#define DECLARE_ONE(name) decltype(&name) name;
  IBV_APIS(DECLARE_ONE)

#undef DECLARE_ONE
#undef IBV_APIS
};

}  // namespace oneflow

#endif  // WITH_RDMA
