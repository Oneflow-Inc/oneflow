#include "node.h"
#include "glog/logging.h"

namespace oneflow {

int32_t NewNodeId() {
  static int32_t node_id_cnt = 0;
  return node_id_cnt++;
}

} // namespace oneflow
