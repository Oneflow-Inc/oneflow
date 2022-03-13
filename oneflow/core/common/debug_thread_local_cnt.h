#ifndef ONEFLOW_CORE_COMMON_DEBUG_THREAD_LOCAL_CNT_H_
#define ONEFLOW_CORE_COMMON_DEBUG_THREAD_LOCAL_CNT_H_

namespace oneflow {
  inline int* MutDebugThreadLocalCnt() {
    static thread_local int debug_cnt;
    return &debug_cnt;
  }
} // namespace oneflow

#endif