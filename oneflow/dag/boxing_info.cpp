#include "dag/boxing_info.h"
#include <glog/logging.h>

namespace oneflow {
void BoxingInfoElement::SetInputs(
  const std::vector<std::string>& inputs) {
  in_num_ = inputs.size();
  int32_t order = 0;
  for (auto& input : inputs) {
    ordered_ins_.push_back(input);
    in_to_order_.insert({ input, order });
    ++order;
  }
}

void BoxingInfoElement::SetOutputs(
  const std::vector<std::string>& outputs) {
  out_num_ = outputs.size();
  int32_t order = 0;
  for (auto& output : outputs) {
    ordered_outs_.push_back(output);
    out_to_order_.insert({ output, order });
    ++order;
  }
}

void BoxingInfoElement::UpdateInput(const std::string& old_in,
  const std::string& new_in) {
  CHECK(in_to_order_.count(old_in) > 0);
  int32_t order = in_to_order_[old_in];
  ordered_ins_[order] = new_in;
  in_to_order_.insert({ new_in, order });
  in_to_order_.erase(old_in);
}

void BoxingInfoElement::UpdateOutput(const std::string& old_out,
  const std::string& new_out) {
  CHECK(out_to_order_.count(old_out) > 0);
  int32_t order = out_to_order_[old_out];
  ordered_outs_[order] = new_out;
  out_to_order_.insert({ new_out, order });
  out_to_order_.erase(old_out);
}

std::vector<std::string> BoxingInfoElement::GetOrderedInputs() const {
  return ordered_ins_;
}

std::vector<std::string> BoxingInfoElement::GetOrderedOutputs() const{
  return ordered_outs_;
}

int32_t BoxingInfoElement::GetOutputOrder(const std::string& output) const {
  auto order_it = out_to_order_.find(output);
  CHECK(order_it != out_to_order_.end());
  return order_it->second;
}

int32_t BoxingInfoElement::GetInputOrder(const std::string& input) const {
  auto order_it = in_to_order_.find(input);
  CHECK(order_it != in_to_order_.end());
  return order_it->second;
}

std::string BoxingInfoElement::GetInput(int32_t order) const {
  CHECK(order < in_num_);
  return ordered_ins_[order];
}

std::string BoxingInfoElement::GetOutput(int32_t order) const {
  CHECK(order < out_num_);
  return ordered_outs_[order];
}

void SingleSideBoxingInfoElement::SetOps(const std::vector<std::string>& ops) {
  op_num_ = ops.size();
  int32_t order = 0;
  for (auto& op : ops) {
    ordered_ops_.push_back(op);
    op_to_order_.insert({ op, order });
    ++order;
  }
}

int32_t SingleSideBoxingInfoElement::GetOpOrder(const std::string& op) const {
  auto order_it = op_to_order_.find(op);
  CHECK(order_it != op_to_order_.end());
  return order_it->second;
}

void BoxingInfo::AddSegmentPairBoxingInfo(
  const SegmentSegmentPair& segment_pair,
  const BoxingInfoElement& boxing_info_element) {
  CHECK(segment_pair_to_boxing_info_.count(segment_pair) == 0);
  segment_pair_to_boxing_info_.insert({ segment_pair, boxing_info_element });

  auto first_segment = segment_pair.first;
  auto second_segment = segment_pair.second;
  AddInputSingleSideInfo(first_segment, boxing_info_element);
  AddOutputSingleSideInfo(second_segment, boxing_info_element);
}

void BoxingInfo::AddInputSingleSideInfo(const std::string& first_segment,
  const BoxingInfoElement& boxing_info_elem) {
  auto info_it = first_segment_to_single_side_info_.find(first_segment);
  if (info_it == first_segment_to_single_side_info_.end()) {
    auto inputs = boxing_info_elem.GetOrderedInputs();
    SingleSideBoxingInfoElement single_side_info;
    single_side_info.SetOps(inputs);
    first_segment_to_single_side_info_.insert(
    { first_segment, single_side_info });
    for (auto& input : inputs) {
      producer_to_first_segment_.insert({ input, first_segment });
    }
  } else {
    // There are multiple segments succeeding |first_segment|, ensure the
    // number of all the boxing_info_element's inputs is equal.
    int32_t in_num = info_it->second.op_num();
    CHECK(in_num == boxing_info_elem.in_num());
  }
}

void BoxingInfo::AddOutputSingleSideInfo(const std::string& second_segment,
  const BoxingInfoElement& boxing_info_elem) {
  auto info_it = second_segment_to_single_side_info_.find(second_segment);
  if (info_it == second_segment_to_single_side_info_.end()) {
    auto outputs = boxing_info_elem.GetOrderedOutputs();
    SingleSideBoxingInfoElement single_side_info;
    single_side_info.SetOps(outputs);
    second_segment_to_single_side_info_.insert(
    { second_segment, single_side_info });
    for (auto& output : outputs) {
      consumer_to_second_segment_.insert({ output, second_segment });
    }
  } else {
    // There are multiple segments preceding |second_segment|, ensure the 
    // number of all the boxing_info_element's outputs is equal.
    int32_t out_num = info_it->second.op_num();
    CHECK(out_num == boxing_info_elem.out_num());
  }
}

std::string BoxingInfo::FirstSegmentFromActorName(
  const std::string& actor) const {
  auto first_segment_it = producer_to_first_segment_.find(actor);
  CHECK(first_segment_it != producer_to_first_segment_.end());
  return first_segment_it->second;
}

std::string BoxingInfo::SecondSegmentFromActorName(
  const std::string& actor) const {
  auto second_segment_it = consumer_to_second_segment_.find(actor);
  CHECK(second_segment_it != consumer_to_second_segment_.end());
  return second_segment_it->second;
}

std::vector<SegmentSegmentPair> BoxingInfo::GetSegmentPairs() const {
  std::vector<SegmentSegmentPair> segment_pairs;
  for (auto& pair : segment_pair_to_boxing_info_) {
    segment_pairs.push_back(pair.first);
  }
  return segment_pairs;
}

BoxingInfoElement& BoxingInfo::GetBoxingInfoElement(
  const SegmentSegmentPair& segment_pair) {
  auto boxing_info_elem_it = segment_pair_to_boxing_info_.find(segment_pair);
  CHECK(boxing_info_elem_it != segment_pair_to_boxing_info_.end());
  return boxing_info_elem_it->second;
}

SingleSideBoxingInfoElement& BoxingInfo::GetSingleSideInfoFromFirstSegment(
  const std::string& first_segment) {
  auto single_side_it = first_segment_to_single_side_info_.find(first_segment);
  CHECK(single_side_it != first_segment_to_single_side_info_.end());
  return single_side_it->second;
}

SingleSideBoxingInfoElement& BoxingInfo::GetSingleSideInfoFromSecondSegment(
  const std::string& second_segment) {
  auto single_side_it = second_segment_to_single_side_info_.find(second_segment);
  CHECK(single_side_it != second_segment_to_single_side_info_.end());
  return single_side_it->second;
}

bool BoxingInfoMap::HasBoxingInfo(
  const std::string& boxing_name) const {
  return boxing_to_info_.count(boxing_name) > 0;
}

void BoxingInfoMap::AddBoxingInfo(const std::string& boxing_name,
  const BoxingInfo& boxing_info) {
  CHECK(!HasBoxingInfo(boxing_name));
  boxing_to_info_.insert({ boxing_name, boxing_info });
}

BoxingInfo& BoxingInfoMap::GetBoxingInfo(
  const std::string& boxing_name) {
  auto boxing_info_it = boxing_to_info_.find(boxing_name);
  CHECK(boxing_info_it != boxing_to_info_.end());
  return boxing_info_it->second;
}

}  // namespace oneflow