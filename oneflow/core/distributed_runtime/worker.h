#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_

#include <fstream>

#include "oneflow/core/distributed_runtime/worker_service.pb.h"
#include "oneflow/core/distributed_runtime/grpc_channel_cache.h"

#include "tensorflow/core/lib/core/status.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace grpc {
class ByteBuffer;
}

namespace oneflow {

typedef std::function<void(const ::tensorflow::Status&)> StatusCallback;

class Worker {
 public:
  explicit Worker(GrpcChannelCache* channel_cache);
  ~Worker() {};


  ::tensorflow::Status GetStatus(GetStatusRequest* request,
                                 GetStatusResponse* response);

  ::tensorflow::Status GetMachineDesc(GetMachineDescRequest* request,
                                      GetMachineDescResponse* response);

  ::tensorflow::Status GetMemoryDesc(GetMemoryDescRequest* request,
                                     GetMemoryDescResponse* response);

  ::tensorflow::Status SendTaskGraph(SendTaskGraphRequest* request,
                                     SendTaskGraphResponse* response);

  ::tensorflow::Status SendMessageAsync(SendMessageRequest* request,
                                   SendMessageResponse* response);

  ::tensorflow::Status ReadDataAsync(ReadDataRequest* request,
                                     ReadDataResponse* response,
                                     StatusCallback done);

  template <typename ProtoMessage>
  void ParseToProto(ProtoMessage& proto_type, std::string& file_name);

  private:
   GrpcChannelCache* channel_cache_;
   struct machine_desc {
     int32_t machine_id;
     std::string ip;
     int32_t port;
   };
   machine_desc machine_desc_;

   struct memory_desc {
     int64_t machine_id;
     int64_t memory_address;
     int64_t remote_token;
   };
   memory_desc memory_desc_;

   std::string machine_desc_file_path;
   std::string memory_desc_file_path; 

};  // GrpcworkerService

}  // namespace oneflow
#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_
