#ifndef _CONTEXT_CONFIG_PARSER_H_
#define _CONTEXT_CONFIG_PARSER_H_
#include <memory>
#include <string>
namespace oneflow {

class SolverDescriptor;
class MachineDescriptor;
class NetDescriptor;
class ResourceDescriptor;
class StrategyDescriptor;

class ConfigParser {
public:
  explicit ConfigParser(const std::string& solver_name);
  ~ConfigParser();

  std::shared_ptr<SolverDescriptor> solver_descriptor() const;
  std::shared_ptr<MachineDescriptor> machine_descriptor() const;
  std::shared_ptr<NetDescriptor> net_descriptor() const;
  std::shared_ptr<ResourceDescriptor> resource_descriptor() const;
  std::shared_ptr<StrategyDescriptor> strategy_descriptor() const;
  void set_strategy_descriptor(
    std::shared_ptr<StrategyDescriptor> strategy_descriptor);
  void Validate();

private:
  std::shared_ptr<SolverDescriptor> solver_descriptor_;
  std::shared_ptr<MachineDescriptor> machine_descriptor_;
  std::shared_ptr<NetDescriptor> net_descriptor_;
  std::shared_ptr<ResourceDescriptor> resource_descriptor_;
  std::shared_ptr<StrategyDescriptor> strategy_descriptor_;

  // Ensure the device set in PlacementGroup consistent with that in Resource
  // configuration
  void CheckDeviceNumInGroup();

  // Ensure the order of layers in group consistent with Net configuration
  void CheckLayerOrderInGroup();

  // Ensure the group use devices in the unit of nodes
  // DO NOT use part of devices in a node for a particular group
  // void CheckDeviceSetInGroup();

  // Ensure the order of group consistent with the Net configuration
  void CheckGroupOrderInStrategy();

  // Ensure the parallelization strategy produce a balanced workload for each 
  // device
  void CheckBalancedWorkLoadInStrategy();

  // TODO(jiyuan): check no overlap between group

  ConfigParser(const ConfigParser& other) = delete;
  ConfigParser& operator=(const ConfigParser& other) = delete;
};
}  // namespace oneflow
#endif  // _CONTEXT_CONFIG_PARSER_H_
