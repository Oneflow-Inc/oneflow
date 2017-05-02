#include "network/rdma/request_pool.h"


namespace oneflow {

RequestPool::RequestPool() : sequence_num (0) {
  msg_pool_.reset(new MessagePool<Message>(kBufferSize));
}

RequestPool::~RequestPool() {
  for (auto& pair : request_dict_) {
    // Firstly, release the |registered_message| of this request.
    delete pair.second->registered_message;
    pair.second->registered_message = nullptr;
    // Secondly, release the request itself.
    delete pair.second;
  }
  // There are also some registered_message in |msg_pool_|, however, which will
  // be released in the desctructor of MessagePool.
}

Request* RequestPool::AllocRequest(bool is_send) {
  int32_t time_stamp = new_time_stamp();
  Request* request = new Request();
  request->time_stamp = time_stamp;
  request->is_send = is_send;
  request->registered_message = msg_pool_->Alloc();
  request_dict_.insert({ time_stamp, request });
  return request;
}

void RequestPool::ReleaseRequest(int32_t time_stamp) {
  auto request = GetRequest(time_stamp);
  // Return the registered message to |msg_pool_|
  msg_pool_->Free(request->registered_message);
  // Destroy the Request object
  delete request;
  // Erase the pair indexed by |time_stamp|
  request_dict_.erase(time_stamp);
}

Request* RequestPool::GetRequest(int32_t time_stamp) const {
  auto request_it = request_dict_.find(time_stamp);
  return request_it->second;
}

Request* RequestPool::UpdateTimeStampAndReuse(int32_t time_stamp) {
  Request* request = GetRequest(time_stamp);
  int32_t new_ts = new_time_stamp();
  request->time_stamp = new_ts;
  request_dict_.erase(time_stamp);
  request_dict_.insert({ new_ts, request });
  return request;
}

int32_t RequestPool::new_time_stamp() {
  int32_t time_stamp = sequence_number_++;
  // Restart like the TCP sequence number
  if (sequence_number_ == std::numeric_limits<int32_t>::max()) {
    sequence_number_ = 0;
  }
  return time_stamp;
}

} // namespace oneflow
