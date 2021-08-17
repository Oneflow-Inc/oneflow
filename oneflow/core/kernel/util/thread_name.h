#include <string>

namespace oneflow {
namespace internal {

void setThreadName(std::string name);

void NUMABind(int numa_node_id); // not enable numa by default

} // namespace internal
} // namespace oneflow
