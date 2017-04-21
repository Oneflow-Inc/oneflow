#include "rdma_wrapper.h"

#include <vector>
/* Windows header file.
 * #include <ws2tcpip.h>
 * #include <iphlpapa.h>
 */

#include "network/network.h"
#include "network/rdma/agency.h"
#include "network/rdma/connection_pool.h"
#include "network/rdma/message_pool.h"
#include "network/rdma/request_pool.h"

namespace oneflow {

namespace {

sockaddr_in GetAddress(const uint64_t* machine_id, int port) {
    sockaddr_in addr = sockaddr_in();
    std::memset(&addr, 0, sizeof(sockaddr_in));
    //TODO() get url by machine_id.
    inet_pton(AF_INET, url, &addr.sin_addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u_short>(port));
    return addr;
}

} // namespace

void RdmaWrapper::Init(uint64_t my_machine_id, 
                       const NetworkTopology& net_topo) {
    my_machine_id_ = my_machine_id;
    net_topology_ = net_topo;
    request_pool_.reset(new RequestPool());
    connection_pool_.reset(new ConnectionPool());

    //NdspiV2Open();
    
    CreateCompletionQueues();
    StartListen();
    EstablishConnection();
}

// |msg| contains src machine_id and dst machine_id 
bool RdmaWrapper::Send(const NetworkMessage& msg) {
    uint64_t dst_machine_id = msg.dst_machine_id;
    Connection* conn = connection_pool_->GetConnection(dst_machine_id);

    // 1. New network request, generating timestamp, get message memory 
    // from message pool.
    Request* send_request = request_pool_->AllocRequest(true);
    // check(send_request)

    // 2. Get network message buffer, copy the message.
    send_request->registered_message->mutable_msg() = msg;

    // NOTE(feiga): The last flag of Send, NO_OP_FLAG_SILENT_SUCCESS means the 
    // successful sending will not generate an event in completion queue.
    // Change the flag to 0 if this does not suit for out design.
    // NOTE(jiyuan): We need to know the completion event of Send to recycle the
    // buffer of message.
    
    // HRESULT hr = conn->queue_pair->Send(
    //         &send_request->time_stamp,
    //         static_cast<const ND2_SGE*>(
    //                 send_request->registered_message->net_memory()->sge()),
    //         1,
    //         0);    

    // CHECK(!FAILED(hr)) << "Failed to send\n";

    return true;
}

void RdmaWrapper::Read(MemoryDescriptor* src, NetworkMemory* dst) {
    Connection* conn = connection_pool_->GetConnection(src->my_machine_id);
    // CHECK

    Request* read_request = request_pool_->AllocRequest(true);
    // CHECK

    // Memorize the Request object for the most recently issued Read verb.
    time_stamp_of_last_read_request = read_request->time_stamp;
    // The read_request->registered_message->mutable_msg() will be set in 
    // |RegisterEventMessage|.

    Memory* dst_memory = reinterpret_cast<Memory*>(dst);
    
    // HRESULT hr = conn->queue_pair->Read(&read_request->time_stamp,
    //         static_cast<const ND2_SGE*>(dst_memory->sge()),
    //         1,
    //         src->address,
    //         src->remote_token,
    //         0);

    // CHECK
}

void RdmaWrapper::Barrier() {
    //TODO() OneFlowWin
}

NetworkMemory* RdmaWrapper::NewNetworkMemory() {
    // TODO() OneFlowWin
    // new MemoryRegion

    // mv new memory_region to Memory
    Memory* memory = new Memory(memory_region);
    return memory;
}

void RdmaWrapper::RegisterEventMessage(MsgPtr event_msg) {
    Request* last_read_request = request_pool_->GetRequest(
            time_stamp_of_last_read_request_);
    last_read_request->registered_message->multable_msg().event_msg = (*event_msg);
}

bool RdmaWrapper::Poll(NetworkResult* result) {
    return PollRecvQueue(result) || PollSendQueue(result);
}

bool RdmaWrapper::PollRecvQueue(NetworkResult* result) {
    // Result r;
    uint32_t len = recv_cq_->GetResults(&r, 1);
    if (len == 0)
        return false;

    // CHECK
    // CHECK

    result->type = NetworkResultType::NET_RECEIVE_MSG;
    // The context is the message timestamp in Recv Request.
    int32_t time_stamp = *(static_cast<int32_t*>(r.RequestContext));
    Request* request = request_pool_->GetRequest(time_stamp);
    // CHECK request

    result->net_msg = request->registered_message->msg();

    // Equivalent to:
    //  request_pool_->ReleaseRequest(time_stamp);
    //  PostReceiveRequest(result->net_msg.src_rank);
    // but is more efficient
    ReRecvRequest(result->net_msg.src_rank, time_stamp);
    return true;
}

bool RdmaWrapper::PollSendQueue(NetworkResult* result) {


}

void RdmaWrapper::PostRecvRequest(uint64_t peer_machine_id) {


}

void RdmaWrapper::RePostRecvRequest(uint64_t peer_machine_id, 
                                    int32_t time_stamp) {


}

const MemoryDescriptor& RdmaWrapper::GetMemoryDescriptor(
        int64_t register_id) const {

}



void RdmaWrapper::CreateCompletionQueues() {


}

void RdmaWrapper::StartListen() {


}

void RdmaWrapper::EstablishConnection() {


}

Connection* RdmaWrapper::NewConnection() {


}

bool RdmaWrapper::TryConnectTo(uint64_t peer_machine_id) {


}

void RdmaWrapper::CompleteConnectionTo(uint64_t peer_machine_id) {


}

// TODO()
// int32_t WaitForConnectionFrom()

void RdmaWrapper::Finalize() { }

} // namespace oneflow

