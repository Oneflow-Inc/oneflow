#include <stdio.h>
#include <infiniband/verbs.h>
#include <iostream>
#include "oneflow/core/dl/include/ibv.h"

using namespace oneflow;
void test() {
  struct ibv_device** dev_list = NULL;
  dev_list = ibv::wrapper.ibv_get_device_list(NULL);
  struct ibv_context* ctx = ibv::wrapper.ibv_open_device(*dev_list);
  ibv_pd* pd = ibv::wrapper.ibv_alloc_pd(ctx);
  char* ib_buf;
  size_t ib_buf_size;
  ib_buf_size = 100;
  ibv_mr* cur_mr = ibv::wrapper.ibv_reg_mr_(pd, ib_buf, ib_buf_size, 0);
  ibv_port_attr port_attr{};
  int ret = 0;
  int IB_PORT = 100;
  ret = ibv::wrapper.ibv_query_port_(ctx, IB_PORT, &port_attr);
  std::cerr << "ret: " << ret << "\n";
  std::cerr << "test done\n";
}

int main() {
  test();
  return 0;
}
