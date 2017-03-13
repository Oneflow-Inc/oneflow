#ifndef _DAG_BOXING_INFO_H_
#define _DAG_BOXING_INFO_H_
#include <unordered_map>
#include <string>
#include <cstdint>
#include "common/string_pair.h"
namespace caffe {
using SegmentSegmentPair = StringPair;
class BoxingInfoElement {
public:
  BoxingInfoElement() = default;
  ~BoxingInfoElement() = default;

  void SetPipeToPipe(bool pipe_to_pipe) { pipe_to_pipe_ = pipe_to_pipe; }
  void SetInputs(const std::vector<std::string>& inputs);
  void SetOutputs(const std::vector<std::string>& outputs);

  void UpdateInput(const std::string& old_in, const std::string& new_in);
  void UpdateOutput(const std::string& old_out, const std::string& new_out);

  int32_t in_num() const { return in_num_; }
  int32_t out_num() const { return out_num_; }
  bool pipe_to_pipe() const { return pipe_to_pipe_; }
  std::vector<std::string> GetOrderedInputs() const;
  std::vector<std::string> GetOrderedOutputs() const;
  int32_t GetOutputOrder(const std::string& output) const;
  int32_t GetInputOrder(const std::string& input) const;
  std::string GetInput(int32_t order) const;
  std::string GetOutput(int32_t order) const;

private:
  bool pipe_to_pipe_{ false };
  int32_t in_num_;
  int32_t out_num_;
  std::vector<std::string> ordered_ins_;
  std::unordered_map<std::string, int32_t> in_to_order_;
  std::vector<std::string> ordered_outs_;
  std::unordered_map<std::string, int32_t> out_to_order_;
};

class SingleSideBoxingInfoElement {
public:
  SingleSideBoxingInfoElement() = default;
  ~SingleSideBoxingInfoElement() = default;

  void SetOps(const std::vector<std::string>& ops);
  int32_t op_num() const { return op_num_; }
  int32_t GetOpOrder(const std::string& op) const;
private:
  int32_t op_num_;
  std::vector<std::string> ordered_ops_;
  std::unordered_map<std::string, int32_t> op_to_order_;
};

class BoxingInfo {
public:
  // BoxingInfo() = default;
  BoxingInfo(bool is_in_boxing) : is_in_boxing_(is_in_boxing) {}
  ~BoxingInfo() = default;

  bool is_in_boxing() const { return is_in_boxing_; }

  void AddSegmentPairBoxingInfo(const SegmentSegmentPair& segment_pair,
    const BoxingInfoElement& boxing_info_element);

  std::vector<SegmentSegmentPair> GetSegmentPairs() const;

  BoxingInfoElement& GetBoxingInfoElement(
    const SegmentSegmentPair& segment_pair);

  SingleSideBoxingInfoElement& GetSingleSideInfoFromFirstSegment(
    const std::string& first_segment);

  SingleSideBoxingInfoElement& GetSingleSideInfoFromSecondSegment(
    const std::string& second_segment);

  std::string FirstSegmentFromActorName(const std::string& actor) const;

  std::string SecondSegmentFromActorName(const std::string& actor) const;
private:
  bool is_in_boxing_{ false };
  std::unordered_map<SegmentSegmentPair, BoxingInfoElement>
    segment_pair_to_boxing_info_;

  // Single side boxing info for out_boxing.
  std::unordered_map<std::string, SingleSideBoxingInfoElement>
    first_segment_to_single_side_info_;

  std::unordered_map<std::string, std::string> producer_to_first_segment_;

  // Single side boxing info for out_boxing.
  std::unordered_map<std::string, SingleSideBoxingInfoElement>
    second_segment_to_single_side_info_;

  std::unordered_map<std::string, std::string> consumer_to_second_segment_;

  void AddOutputSingleSideInfo(const std::string& second_segment,
    const BoxingInfoElement& boxing_info_elem);
  void AddInputSingleSideInfo(const std::string& first_segment,
    const BoxingInfoElement& boxing_info_elem);
};

class BoxingInfoMap {
public:
  BoxingInfoMap() = default;
  ~BoxingInfoMap() = default;
  bool HasBoxingInfo(const std::string& boxing_name) const;
  void AddBoxingInfo(const std::string& boxing_name,
    const BoxingInfo& boxing_info);
  BoxingInfo& GetBoxingInfo(const std::string& boxing_name);

private:
  std::unordered_map<std::string, BoxingInfo> boxing_to_info_;
};

}  // namespace caffe
#endif  // _DAG_BOXING_INFO_