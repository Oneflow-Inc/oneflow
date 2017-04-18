#ifndef ONEFLOW_CONTEXT_ID_MAP_H_
#define ONEFLOW_CONTEXT_ID_MAP_H_
#include <glog/logging.h>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory>
// #include "caffe.pb.h"

namespace oneflow {
/*
Each device has several identities. Either |logical_id| or |device_id| can uniquely
indicate a device in all the available devices on all the allocated machines.

The |logical_id| is assigned to a device according to its occurring order in
Resource protobuf file.

The |device_id| is obtained by combining the device's |machine_id| and |local_id|
through some bit operations.

We firstly assign a |machine_id| to each machine according to its order in the
Resource protobuf file, and then assign a |local_id| to each device on a particular
machine according to its order in Resource protobuf file confined to the residing
machine. The |device_id| is obtained by placing |machine_id| at some most
significant bits and placing |local_id| at some least significant bits. Note that
in most cases, the |logical_id| is not the same as the |device_id|.
Both |local_id| and |physical_id| are meaningful only locally to a particular machine.
|local_id| is a logical local id, while |physical_id| is an id recognized by
CUDA driver (e.g., SetDeviceId()).
Let's take a look at a toy example. We have 3 machines, each machine has 4 devices.
However, the system only allocates 3 devices to this job.
The possible IDs could be:

logical_id    machine_id    local_id    device_id    physical_id
0             0x00          0x00        0x00         0
1             0x00          0x01        0x01         2
2             0x00          0x02        0x02         3
3             0x01          0x00        0x10         1
4             0x01          0x01        0x11         2
5             0x01          0x02        0x12         3
6             0x02          0x00        0x20         0
7             0x02          0x01        0x21         1
8             0x02          0x02        0x22         2

In the above example, we use 4-bit to represent machine_id and another 4-bit for
representing local_id.

NOTE(jiyuan):
1, If not explicitly stated otherwise, we use device_id since it is unique and
containing the residing machine's id.
2, Ensure:
    thread_local_id == device_local_id
    thread_id == device_id
*/

// TODO(jiyuan): we should define some aliases for various types of ID.
// For example, register_id may need int64_t, maybe we declare a RegisterIdType
// as the type of register_id.

class ConfigParser;

class IDMap {
  struct DeviceInfo {
    int32_t logical_id;
    int32_t device_id;
    int32_t machine_id;
    int32_t local_id;
    int32_t physical_id;
  };

 public:
  // explicit IDMap(std::shared_ptr<ConfigParser> config);
  IDMap();
  ~IDMap();

  int32_t machine_num() const;
  int32_t device_num_each_machine() const;
  int32_t total_device_num() const;
  bool is_device_thread(int32_t thread_local_id) const;

  int64_t data_id_from_device_and_piece(
    int32_t device_id, int64_t piece_id) const;
  int32_t device_id_from_data_id(int64_t data_id) const;
  int64_t piece_id_from_data_id(int64_t data_id) const;

  int32_t machine_id_from_logical_id(int32_t logical_id) const;
  int32_t device_id_from_logical_id(int32_t logical_id) const;
  int32_t logical_id_from_device_id(int32_t device_id) const;

  int32_t machine_id_from_device_id(int32_t device_id) const;
  int32_t local_id_from_device_id(int32_t device_id) const;
  int32_t physical_id_from_device_id(int32_t device_id) const;
  int32_t device_id_from_machine_and_local(
    int32_t machine_id, int32_t local_id) const;

  // Specific to current node
  int32_t physical_id_from_local_id(int32_t local_id) const;
  int32_t local_id_from_physical_id(int32_t physical_id) const;

  int32_t thread_id_from_machine_and_local(
    int32_t machine_id, int32_t local_id) const;
  int32_t machine_id_from_thread_id(int32_t thread_id) const;
  int32_t local_id_from_thread_id(int32_t thread_id) const;

  int32_t task_id_from_thread_id_and_task_local_id(int32_t thread_id,
    int32_t task_local_id) const;
  int32_t task_local_id_from_task_id(int32_t task_id) const;

  int32_t machine_id_from_register_id(int64_t register_id) const;
  int32_t thread_id_from_register_id(int64_t register_id) const;
  int32_t thread_local_id_from_register_id(int64_t register_id) const;
  int32_t task_id_from_register_id(int64_t register_id) const;
  int32_t task_local_id_from_register_id(int64_t register_id) const;
  int64_t group_id_from_register_id(int64_t register_id) const;
  int32_t group_local_id_from_register_id(int64_t register_id) const;
  int32_t register_local_id_from_register_id(int64_t register_id) const;

  int64_t group_id_from_task_id_and_group_local_id(int32_t task_id,
    int32_t group_local_id) const;
  int64_t register_id_from_group_id_and_register_local_id(int64_t group_id,
    int32_t register_local_id) const;
  int32_t task_id_from_group_id(int64_t group_id) const;

  int32_t thread_id_from_task_id(int32_t task_id) const;
  int32_t thread_local_id_from_task_id(int32_t task_id) const;
  int32_t machine_id_from_task_id(int32_t task_id) const;

  int32_t device_thread_local_id(int32_t local_id) const;
  int32_t data_thread_local_id() const;
  int32_t boxing_thread_local_id() const;
  int32_t net_thread_local_id() const;

  int32_t new_task_local_id(int32_t thread_id);
  int32_t new_group_local_id(int32_t task_id);
  int32_t new_register_local_id(int64_t group_id);

 private:
  std::shared_ptr<ConfigParser> config_parser_;
  int32_t machine_num_;
  int32_t device_num_each_machine_;
  int32_t total_device_num_;
  std::vector<DeviceInfo> devices_info_;
  std::unordered_map<int32_t, int32_t> device2logical_;
  int32_t data_thread_local_id_;
  int32_t boxing_thread_local_id_;
  int32_t net_thread_local_id_;
  std::unordered_map<int32_t, int32_t> thread_id_to_task_local_id_counter_;
  std::unordered_map<int32_t, int32_t> task_id_to_group_local_id_counter_;
  std::unordered_map<int64_t, int32_t>
    group_id_to_register_local_id_counter_;

  // thread ids
  // The order of thread types:
  // device threads, data threads, boxing threads, net threads

  // For 64 bit register id
  //   machine  thread       task             group            register
  // |--------|--------|----------------|----------------|----------------|
  const int32_t machine_bit_num_ = 8;   // maximumly 128 machines
  const int32_t thread_local_bit_num_ = 8;
  const int32_t task_local_bit_num_ = 16;
  const int32_t group_local_bit_num_ = 16;
  const int32_t register_local_bit_num_ = 16;


  const int32_t device_bit_num_ = machine_bit_num_ + thread_local_bit_num_;  // 16;
  const int32_t task_bit_num_ = device_bit_num_ + task_local_bit_num_;  // 32;

  // For 64 bit data id
  //      device                         piece
  // |----------------|------------------------------------------------|
  const int32_t piece_bit_num_ = 64 - device_bit_num_;  //  48;

  const int32_t register_bit_num_
    = group_local_bit_num_ + register_local_bit_num_;  // 32 bits

  // Use unsigned int as mask of bit operation:
  // "<<" operation fills 0 to the right side.
  // ">>" operation fills 0 or copy the sign bit to the left side for signed int
  // ">>" operation fills 0 to the left side for unsigned int
  const uint64_t uint64_full_mask_ = 0xFFFFFFFFFFFFFFFF;
  const uint32_t uint32_full_mask_ = 0xFFFFFFFF;

  int32_t make_thread_id_with_machine_and_local(
    int32_t machine_id,
    int32_t local_id) const;

  IDMap(const IDMap& other) = delete;
  IDMap& operator=(const IDMap& other) = delete;
};

}  // namespace oneflow
#endif  // ONEFLOW_CONTEXT_ID_MAP_H_
