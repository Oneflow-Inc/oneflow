#ifndef ONEFLOW_NETWORK_RDMA_REQUEST_POOL_H_
#define ONEFLOW_NETWORK_RDMA_REQUEST_POOL_H_

#include <unordered_map>
#include <cstdint>
#include <memory>
#include "network/rdma/message_pool.h"
#include "network/rdma/switch.h"


namespace oneflow{

class RegisteredNetworkMessage;

/* move request to system/request.h
struct Request {
    bool is_send;
    int32_t time_stamp;
    RegisteredNetworkMessage* registered_message;
};
*/

class RequestPool {
public:
    RequestPool();
    ~RequestPool();

    // Allocate a Request object from the pool.
    // |is_send| is true for Send request, false for Receive request.
    Request* AllocRequest(bool is_send);

    // Release the Request object indexed by |time_stamp|.
    void ReleaseRequest(int32_t time_stamp);

    // Get the Request object indexed by |time_stamp|.
    Request* GetRequest(int32_t time_stamp) const;

    // Update and reuse the Request object indexed by |time_stamp|, to avoid 
    // unnecessary object destroy and creation. It is useful when only time_stamp
    // needs update, while other properties do not change.
    Request* UpdateTimeStampAndReuse(int32_t time_stamp);

private:
    int32_t sequence_number_;
    std::unordered_map<int32_t, Request*> request_dict_;
    std::shared_ptr<MessagePool<RegisteredNetworkMessage>> msg_pool_;
    static const int32_t kBufferSize = 64;

    int32_t new_time_stamp();

    RequestPool(const RequestPool& other) = delete;
    RequestPool& operator=(const RequestPool& other) = delete;
};

} // namespace oneflow


#endif // ONEFLOW_NETWORK_RDMA_REQUEST_POOL_H_
