#ifndef MASTER_SERVICE_H_
#define MASTER_SERVICE_H_

namespace grpc{
class ServerBuilder;
}

namespace oneflow{

class AsyncServiceInterface;

AsyncServiceInterface* NewGrpcMasterService(::grpc::ServerBuilder* builder);

}
#endif
