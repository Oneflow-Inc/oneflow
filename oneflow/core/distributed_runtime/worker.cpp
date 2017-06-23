#include "oneflow/core/distributed_runtime/worker.h"
#include "oneflow/core/distributed_runtime/grpc_tensor_coding.h"
#include "oneflow/core/distributed_runtime/worker.pb.h"

//#include "oneflow/core/network/network_message.h"
//#include "oneflow/core/network/network_memory.h"

//#include "context/id_map.h"
//#include "oneflow/core/runtime/comm_bus.h"

namespace oneflow {

Worker::Worker(GrpcChannelCache* channel_cache)
  : channel_cache_(channel_cache) {}

::tensorflow::Status Worker::GetStatus(GetStatusRequest* request,
                                       GetStatusResponse* response) {
  std::cout<<"request from client = "<<request->status_test() << std::endl;
  response->set_status_test("get_status_test from server");

  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::GetMachineDesc(GetMachineDescRequest* request,
                                            GetMachineDescResponse* response) {
  response->mutable_machine_desc()->set_machine_id(0);
  response->mutable_machine_desc()->set_ip("192.168.1.11");
  response->mutable_machine_desc()->set_port(50051);
  
  std::cout << "request from client = " << request->machine_desc_test() << std::endl; 
  response->set_machine_desc_test("machine_desc_test from server");

  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::GetMemoryDesc(GetMemoryDescRequest* request,
                                           GetMemoryDescResponse* response) {
  oneflow::MemoryDesc memory_desc_for_resp;
  memory_desc_file_path = "./memory_desc.txt";
  ParseToProto(memory_desc_for_resp, memory_desc_file_path);
  response->mutable_memory_desc()->set_machine_id(memory_desc_for_resp.machine_id());
  response->mutable_memory_desc()->set_memory_address(memory_desc_for_resp.memory_address());
  response->mutable_memory_desc()->set_remoted_token(memory_desc_for_resp.remoted_token());

  std::cout << "request from client = " << request->memory_desc_test() << std::endl;
  response->set_memory_desc_test("memory_desc_test from server");
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::SendTaskGraph(SendTaskGraphRequest* request,
                                           SendTaskGraphResponse* response) {
  //TODO
  //convert TaskGraphDef to TaskGraph
  //then init thread to start executor taskgraph
  //response is empty
  std::cout << "SendTaskGraph request from client = " << request->send_task_graph_test() << std::endl;
  response->set_send_task_graph_test("SendTaskGraph from server");
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::SendMessageAsync(SendMessageRequest* request,
                                         SendMessageResponse* response) {
  /*
  NetworkMessage* network_message = new NetworkMessage;
  std::shared_ptr<EventMessage> event_message;
  event_message->set_to_task_id(request->network_message().event_message().to_task_id());
  event_message->set_register_id(request->network_message().event_message().register_id());

  IDMap* id_map = new IDMap;
  int64_t thread_local_id 
    = id_map->thread_local_id_from_task_id(event_message->to_task_id_);

  CommBus* comm_bus = new CommBus; 
  comm_bus->queues_[thread_local_id]->Push(event_message);
  */
  std::cout << "SendMessage request from client = " << request->send_message_test() << std::endl;
  response->set_send_message_test("send_message_test from server");

  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::ReadDataAsync(ReadDataRequest* request,
                                           ::grpc::ByteBuffer* response,
                                           StatusCallback done) {
  //const int64_t register_id = request->register_id();
  
  const Tensor val;
  grpc::EncodeTensorToByteBuffer(false, val, response);

  //std::cout << "request from client = " << request->read_data_test() << std::endl;
  //response->set_read_data_test("read_data_test from server");
  return ::tensorflow::Status::OK();
}

template <typename ProtoMessage>
void Worker::ParseToProto(ProtoMessage& proto_type, std::string& file_name) {
  std::ifstream input_file(file_name); 
  google::protobuf::io::IstreamInputStream proto_file(&input_file);
  if(!google::protobuf::TextFormat::Parse(&proto_file, &proto_type)) {
    input_file.close();
  }  // end if
}  // end Parsetoproto

}  // namespace oneflow
