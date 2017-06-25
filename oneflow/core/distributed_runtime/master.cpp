#include "oneflow/core/distributed_runtime/master.h"

namespace oneflow {

Master::Master(GrpcChannelCache* channel_cache)
  : channel_cache_(channel_cache) {}

Master::~Master() {}

::tensorflow::Status Master::SendGraph(SendGraphRequest* request,
                       SendGraphResponse* response) {
  // Barrier();
  std::cout << "Server: request from Client = " << request->tmp() << std::endl;
  response->set_tmp(8);

  return ::tensorflow::Status::OK();
}

void Master::Barrier() {
}

}  // namespace oneflow



