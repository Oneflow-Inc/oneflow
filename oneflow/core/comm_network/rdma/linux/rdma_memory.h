#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_LINUX_RDMA_MEMORY_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_LINUX_RDMA_MEMORY_H 

namespace oneflow{

class RdmaMemDesc {
};
  
class RdmaMem {
public:
  void Register(void* mem_ptr, size_t byte_size);
  void UnRegister();

private:
  void* mem_ptr_;
  size_t byte_size_;
};

}

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_LINUX_RDMA_MEMORY_H
