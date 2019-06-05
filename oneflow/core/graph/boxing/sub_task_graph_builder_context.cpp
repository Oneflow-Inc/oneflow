#include "oneflow/core/graph/boxing/sub_task_graph_builder_context.h"

namespace oneflow {

namespace {

bool IsMemoryCaseEquals(const MemoryCase& lhs, const MemoryCase& rhs) {
  if (lhs.has_host_mem() && rhs.has_host_mem()) {
    return true;
  } else if (lhs.has_device_cuda_mem() && rhs.has_device_cuda_mem()
             && lhs.device_cuda_mem().device_id() == rhs.device_cuda_mem().device_id()) {
    return true;
  } else {
    CHECK(lhs.has_host_mem() || lhs.has_device_cuda_mem());
    CHECK(rhs.has_host_mem() || rhs.has_device_cuda_mem());
    return false;
  }
}

}  // namespace

SubTskGphBuilderCtx::SubTskGphBuilderCtx(TaskGraph* task_graph) : task_graph_(task_graph) {}

TaskGraph* SubTskGphBuilderCtx::task_graph() { return task_graph_; }

TaskNode* SubTskGphBuilderCtx::GetProxyNode(TaskNode* src_node, const MemoryCase& src_mem_case,
                                            int64_t dst_machine_id,
                                            const MemoryCase& dst_mem_case) {
  const auto key = std::make_pair(dst_machine_id, dst_mem_case);
  if (node2proxies_.find(src_node) != node2proxies_.cend()
      && node2proxies_.at(src_node).find(key) != node2proxies_.at(src_node).cend()) {
    return node2proxies_.at(src_node).at(key);
  } else {
    if (dst_machine_id == src_node->machine_id()
        && IsMemoryCaseEquals(dst_mem_case, src_mem_case)) {
      node2proxies_[src_node][key] = src_node;
      return src_node;
    } else if (dst_mem_case.has_device_cuda_mem()) {
      TaskNode* proxy_on_dst_host =
          GetProxyNode(src_node, src_mem_case, dst_machine_id, MakeHostMemCase());
      CopyHdTaskNode* copy_task = task_graph()->NewNode<CopyHdTaskNode>();
      copy_task->Init(CopyHdOpConf::H2D, proxy_on_dst_host->machine_id(),
                      dst_mem_case.device_cuda_mem().device_id());
      Connect<TaskNode>(proxy_on_dst_host, task_graph()->NewEdge(), copy_task);
      node2proxies_[src_node][key] = copy_task;
      return copy_task;
    } else if (dst_mem_case.has_host_mem()) {
      if (src_node->machine_id() == dst_machine_id) {
        if (src_mem_case.has_device_cuda_mem()) {
          CopyHdTaskNode* copy_task = task_graph()->NewNode<CopyHdTaskNode>();
          copy_task->Init(CopyHdOpConf::D2H, src_node->machine_id(),
                          src_mem_case.device_cuda_mem().device_id());
          Connect<TaskNode>(src_node, task_graph()->NewEdge(), copy_task);
          node2proxies_[src_node][key] = copy_task;
          return copy_task;
        } else {
          UNIMPLEMENTED();
        }
      } else {
        TaskNode* proxy_on_src_host =
            GetProxyNode(src_node, src_mem_case, src_node->machine_id(), MakeHostMemCase());
        CopyCommNetTaskNode* copy_comm_net_task = task_graph()->NewNode<CopyCommNetTaskNode>();
        copy_comm_net_task->Init(dst_machine_id, proxy_on_src_host->machine_id());
        Connect<TaskNode>(proxy_on_src_host, task_graph()->NewEdge(), copy_comm_net_task);
        node2proxies_[src_node][key] = copy_comm_net_task;
        return copy_comm_net_task;
      }
    } else {
      UNIMPLEMENTED();
    }
  }
}

}  // namespace oneflow
