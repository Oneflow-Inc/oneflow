#ifndef WORKER_SERVICE_H_
#define WORKER_SERVICE_H_

namespace grpc {
class ServerBuilder;
}

namespace oneflow {
class AsyncServiceInterface;

AsyncServiceInterface* NewGrpcWorkerService(::grpc::ServerBuilder* builder);

}// namespace oneflow

#endif
