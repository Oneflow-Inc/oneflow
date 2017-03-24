#ifndef _CONTEXT_RESOURCE_DESCRIPTOR_H_
#define _CONTEXT_RESOURCE_DESCRIPTOR_H_
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
/*
Parsed content from resource proto.
*/
namespace oneflow {
class ResourceDescriptor {
  struct MachineInfo {
    std::string name;
    std::string port;
    std::vector<int32_t> device_ids;  // Physical device ID
    std::unordered_map<int32_t, int32_t> physical2local;
  };
public:
  explicit ResourceDescriptor(const std::string& resource_name);
  ~ResourceDescriptor();

  int32_t machine_num() const;
  // Return the number of devices allocated to this job on a machine
  int32_t device_num_per_machine() const;
  int32_t total_device_num() const;
  std::string machine_name(int32_t id) const;
  std::string machine_port(int32_t id) const;
  std::vector<int32_t> machine_device_ids(int32_t id) const;
  int32_t local_from_physical(int32_t machine_id, int32_t physical_id) const;

  private:
  int32_t machine_num_;
  int32_t device_num_per_machine_;
  std::vector<MachineInfo> machine_infos_;

  ResourceDescriptor(const ResourceDescriptor& other);
  ResourceDescriptor& operator=(const ResourceDescriptor& other);
};
}  // namespace oneflow
#endif  // _CONTEXT_RESOURCE_DESCRIPTOR_H_
