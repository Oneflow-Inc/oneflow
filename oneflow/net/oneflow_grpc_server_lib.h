#ifndef ONEFLOW_GRPC_SERVER_H_
#define ONEFLOW_GRPC_SERVER_H_

namespace oneflow{

class GrpcServer{
 public:
  GrpcServer();
  ~GrpcServer();
  
  int Init();
};

}//end namespace oneflow

#endif


