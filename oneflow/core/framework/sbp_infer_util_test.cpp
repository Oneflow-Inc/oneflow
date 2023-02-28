/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/framework/nd_sbp.h"

#include <gtest/gtest.h>

namespace oneflow {
namespace test {

namespace {

bool ParseNdSbpSignatureFromString(const std::string& nd_sbp_signature_str,
                                   NdSbpSignature& nd_sbp_signature) {
  auto* bn2nd_sbp = nd_sbp_signature.mutable_bn_in_op2nd_sbp();
  std::string arg_name = "in";
  bool meet_nd_sbp_group = false;
  bool meet_split = false;
  int nd_sbp_group_id = 0;
  std::vector<std::string> nd_sbp_str_group;
  size_t pos = 0;
  while (pos < nd_sbp_signature_str.size()) {
    const char& c = nd_sbp_signature_str[pos];
    pos++;
    if (c == ' ') {
      continue;
    } else if (c == '(') {
      if (!meet_nd_sbp_group) {
        // enter a nd-sbp group
        meet_nd_sbp_group = true;
        nd_sbp_str_group.emplace_back();
        continue;
      } else {
        // meet left parentheses of S(x)
        meet_split = true;
      }
    } else if (c == ')') {
      if (meet_split) {
        // meet right parentheses of S(x)
        meet_split = false;
      } else if (meet_nd_sbp_group) {
        // leave a nd-sbp group
        meet_nd_sbp_group = false;
        std::string bn = arg_name + "_" + std::to_string(nd_sbp_group_id);
        if (!ParseNdSbpFromStringList(nd_sbp_str_group, &(*bn2nd_sbp)[bn])) { return false; }
        nd_sbp_str_group.clear();
        continue;
      } else {
        return false;
      }
    } else if (c == ',') {
      if (meet_nd_sbp_group) {
        nd_sbp_str_group.emplace_back();
      } else {
        nd_sbp_group_id += 1;
      }
      continue;
    } else if (c == '-') {
      if (pos < nd_sbp_signature_str.size() && nd_sbp_signature_str[pos] == '>') {
        // in args parsing has finished, parse out args
        arg_name = "out";
        nd_sbp_group_id = 0;
        // skip '>' in substr '->'
        pos++;
        continue;
      } else {
        return false;
      }
    } else {
      // do nothing
    }
    nd_sbp_str_group.back() += c;
  }
  return true;
}

std::string NdSbpSignature2String(const NdSbpSignature& nd_sbp_signature,
                                  const std::vector<std::string>& inputs,
                                  const std::vector<std::string>& outputs) {
  std::ostringstream ss;
  auto BnNdSbpToString = [&](const std::string& bn) {
    auto iter = nd_sbp_signature.bn_in_op2nd_sbp().find(bn);
    CHECK(iter != nd_sbp_signature.bn_in_op2nd_sbp().end());
    ss << NdSbpToString(iter->second);
  };
  auto ArgsNdSbpToString = [&](const std::vector<std::string>& arg_bns) {
    for (size_t i = 0; i < arg_bns.size(); ++i) {
      if (i > 0) { ss << ", "; }
      BnNdSbpToString(arg_bns[i]);
    }
  };
  ArgsNdSbpToString(inputs);
  ss << " -> ";
  ArgsNdSbpToString(outputs);
  return ss.str();
}

void TestDeduplicateNdSbpSignature(const std::vector<std::string>& nd_sbp_signature_str_list,
                                   const std::vector<std::string>& input_bns,
                                   const std::vector<std::string>& output_bns) {
  // parse
  std::vector<NdSbpSignature> nd_sbp_sig_list;
  nd_sbp_sig_list.reserve(nd_sbp_signature_str_list.size());
  for (const auto& nd_sbp_signature_str : nd_sbp_signature_str_list) {
    nd_sbp_sig_list.emplace_back();
    ASSERT_TRUE(ParseNdSbpSignatureFromString(nd_sbp_signature_str, nd_sbp_sig_list.back()));
  }

  // shuffle and repeat
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(nd_sbp_sig_list.begin(), nd_sbp_sig_list.end(), gen);
  nd_sbp_sig_list.reserve(nd_sbp_sig_list.size() + nd_sbp_sig_list.size() / 2);
  std::copy_n(nd_sbp_sig_list.begin(), nd_sbp_sig_list.size() / 2,
              std::back_inserter(nd_sbp_sig_list));
  std::shuffle(nd_sbp_sig_list.begin(), nd_sbp_sig_list.end(), gen);

  // dedup and sort
  std::vector<std::string> bns;
  bns.insert(bns.end(), input_bns.begin(), input_bns.end());
  bns.insert(bns.end(), output_bns.begin(), output_bns.end());
  DeduplicateNdSbpSignatureList(&nd_sbp_sig_list, bns);

  // compare
  ASSERT_EQ(nd_sbp_signature_str_list.size(), nd_sbp_sig_list.size());
  for (size_t i = 0; i < nd_sbp_sig_list.size(); ++i) {
    auto nd_sbp_sig_result = NdSbpSignature2String(nd_sbp_sig_list[i], input_bns, output_bns);
    ASSERT_EQ(nd_sbp_sig_result, nd_sbp_signature_str_list[i]);
  }
}

}  // namespace

TEST(SbpInferUtil, DeduplicateNdSbpSignatureList) {
  TestDeduplicateNdSbpSignature(
      {
          "(B, B) -> (B, B)",
          "(B, P) -> (B, P)",
          "(B, S(0)) -> (B, S(0))",
          "(B, S(1)) -> (B, S(1))",
          "(B, S(3)) -> (B, S(2))",
          "(P, B) -> (P, B)",
          "(P, P) -> (P, P)",
          "(P, S(0)) -> (P, S(0))",
          "(P, S(1)) -> (P, S(1))",
          "(P, S(3)) -> (P, S(2))",
          "(S(0), B) -> (S(0), B)",
          "(S(0), P) -> (S(0), P)",
          "(S(0), S(0)) -> (S(0), S(0))",
          "(S(0), S(1)) -> (S(0), S(1))",
          "(S(0), S(3)) -> (S(0), S(2))",
          "(S(1), B) -> (S(1), B)",
          "(S(1), P) -> (S(1), P)",
          "(S(1), S(0)) -> (S(1), S(0))",
          "(S(1), S(1)) -> (S(1), S(1))",
          "(S(1), S(3)) -> (S(1), S(2))",
          "(S(3), B) -> (S(2), B)",
          "(S(3), P) -> (S(2), P)",
          "(S(3), S(0)) -> (S(2), S(0))",
          "(S(3), S(1)) -> (S(2), S(1))",
          "(S(3), S(3)) -> (S(2), S(2))",
      },
      {"in_0"}, {"out_0"});
}

}  // namespace test
}  // namespace oneflow
