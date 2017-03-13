#ifndef _REGISTER_INFO_H_
#define _REGISTER_INFO_H_
#include <vector>
#include <unordered_set>
#include <string>
#include <memory>
#include <unordered_map>
#include "common/shape.h"
#include "memory/memory_manager.h"
/*
Contain all the required info needed for creating a new register.
*/
namespace caffe {
enum class RegisterType {
  kDataType = 0,
  kDataDiffType,
  kModelType,
  kModelDiffType
};

enum class EnvelopeFlag {
  kInEnvelope = 0,
  kOutEnvelope,
  kOnEnvelope
};

class RegisterInfo {
 public:
  RegisterInfo();
  RegisterInfo(int64_t group_id);  // Default constructor, fields set later
  RegisterInfo(RegisterType register_type, DeviceType device_type,
    int64_t group_id, bool network);
  ~RegisterInfo() = default;

  RegisterInfo(const RegisterInfo& other) = default;
  RegisterInfo& operator=(const RegisterInfo& other) = default;

  void set_register_type(RegisterType type);
  void set_device_type(DeviceType type);
  void set_group_id(int64_t group_id);
  void set_network(bool network);

  RegisterType register_type() const;
  DeviceType device_type() const;
  int64_t group_id() const;
  bool network() const;
  bool has_envelope() const;

  int64_t aligned_memory_needed() const;
  int64_t total_element_num() const;

  const std::vector<std::string>& GetAllBlobNames() const;

  // The following three getters can be called even without being finalized. 
  // Useful while merging two incomplete RegisterInfos.
  const std::vector<std::string>& GetBlobNamesInsideEnvelope() const;
  const std::vector<std::string>& GetBlobNamesOutsideEnvelope() const;
  std::vector<std::string> GetEnvelopeNames() const;

  Shape GetEnvelopeShape() const;
  // Query the shape of a blob according to its name. You can get the envelope
  // shape (if the register has one) when |blob_name| is the envelope_name_.
  const Shape& GetBlobShape(const std::string& blob_name) const;

  // |type| is used to indicate whether this blob needs to be included
  // in an envelope blob, outside the envelope blob or itself is an envelope blob
  void AddEmptyBlob(const std::string& blob_name, EnvelopeFlag flag);
  void SetBlobShape(const std::string& blob_name, const Shape& shape);

  // NOTE(jiyuan): ensure to call |Finalize| to get a complete invariant.
  void Finalize();

 private:
  RegisterType register_type_;
  DeviceType device_type_;
  int64_t group_id_;
  bool has_envelope_;
  bool network_;
  bool is_finalized_;

  std::vector<std::string> blob_names_;
  std::unordered_set<std::string> blob_name_set_;
  std::unordered_map<std::string, Shape> blob_dict_;

  // Taking blobs in types of either kModel or kModelDiff as examples, marking
  // them as envelope blobs will ensure the blobs are in a continuous memory 
  // region ordered according to the names in the vector.
  std::vector<std::string> blob_names_inside_envelope_;

  // |envelope_shape_| is used to describe the virtual blob formed by
  // accumulating all the blobs in envelope together.
  std::vector<std::string> envelope_names_;
  Shape envelope_shape_;

  // For blobs without envelope's continuous memory region guarantee.
  std::vector<std::string> blob_names_outside_envelope_;

  int64_t unaligned_memory_needed_;
  int64_t unaligned_memory_needed_for_envelope_;
  int64_t unaligned_memory_needed_for_non_envelope_;
  int64_t aligned_memory_needed_;
  int64_t total_elem_num_;
};
}  // namespace caffe
#endif  // _REGISTER_INFO_H_
