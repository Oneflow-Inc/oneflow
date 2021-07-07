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

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/ctc_beam_search_decoder.h"

namespace oneflow {
namespace {

template<typename T>
struct CTCBeamSearchDecoderFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, int64_t* decoded_ptr, T* log_probability_ptr,
                  const T* log_probs_ptr, const int64_t* input_lengths_ptr,
                  const int32_t beam_width, const int32_t top_paths, const int64_t max_input_length,
                  const int64_t batch_size, const int64_t num_labels) {
    const T prune_threshold = 0.001;
    FOR_RANGE(int64_t, b, 0, batch_size) { CHECK_GE(max_input_length, input_lengths_ptr[b]); }
    NdIndexOffsetHelper<int64_t, 3> input_helper(max_input_length, batch_size, num_labels);
    // decoded_ptr [batch_size, max_input_length]

    FOR_RANGE(int64_t, b, 0, batch_size) {
      const int32_t input_length = input_lengths_ptr[b];
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      std::multimap<float, std::vector<int32_t>, std::greater<float>> A_next_inv;
      // For a given time step, Pb maps prefixes to the probability of all
      // candidate sequences that end in a blank and Pnb maps prefixes to the
      // probability of all candidate sequences that don't end in a blank.
      std::vector<std::map<std::vector<int32_t>, float>> Pb(
          input_length + 1, std::map<std::vector<int32_t>, float>());
      std::vector<std::map<std::vector<int32_t>, float>> Pnb(
          input_length + 1, std::map<std::vector<int32_t>, float>());
      std::set<std::vector<int32_t>> A_prev;
      Pb[0][std::vector<int32_t>()] = 1;
      Pnb[0][std::vector<int32_t>()] = 0;
      A_prev.insert(std::vector<int32_t>());

      for (int t = 0; t < input_length; t++) {
        const T* prob_data_t = &log_probs_ptr[input_helper.NdIndexToOffset(t, b, 0)];
        std::vector<int32_t> pruned_alpha;
        for (int32_t c = 0; c < num_labels; c++) {
          if (prob_data_t[c] > prune_threshold) { pruned_alpha.push_back(c); }
        }

        // If the pruned alphabet is empty, don't use pruning.
        if (pruned_alpha.size() == 0) {
          pruned_alpha = std::vector<int32_t>(num_labels);
          std::iota(pruned_alpha.begin(), pruned_alpha.end(), 0);
        }

        for (auto const& l : A_prev) {
          // We skip the code handling the end character from the article since
          // our system does not support an end character.
          for (auto const c : pruned_alpha) {
            // Assumption: blank character always mapped to index 0
            if (c == 0) {
              Pb[t + 1][l] += prob_data_t[c] * (Pb[t][l] + Pnb[t][l]);
            } else {
              std::vector<int32_t> l_plus = std::vector<int32_t>(l);
              l_plus.push_back(c);
              if (l.size() > 0 && c == l.back()) {
                Pnb[t + 1][l_plus] += prob_data_t[c] * Pb[t][l];
                Pnb[t + 1][l] += prob_data_t[c] * Pnb[t][l];
              } else {
                Pnb[t + 1][l_plus] += prob_data_t[c] * (Pb[t][l] + Pnb[t][l]);
              }

              if (A_prev.find(l_plus) == A_prev.end()) {
                Pb[t + 1][l_plus] += prob_data_t[0] * (Pb[t][l_plus] + Pnb[t][l_plus]);
                Pnb[t + 1][l_plus] += prob_data_t[c] * Pnb[t][l_plus];
              }
            }
          }
        }

        std::map<std::vector<int32_t>, float> A_next(Pb[t + 1]);
        for (const auto& it : Pnb[t + 1]) { A_next[it.first] += it.second; }
        A_next_inv.clear();
        for (const auto& it : A_next) { A_next_inv.insert({it.second, it.first}); }

        A_prev.clear();
        auto it = A_next_inv.begin();
        for (int j = 0; j < beam_width; j++) {
          if (it == A_next_inv.end()) { break; }
          A_prev.insert(it->second);
          it++;
        }
      }

      // const int total_candidates = batch_size * top_paths;
      // auto* output_len =
      //     Output(OUTPUT_LEN, std::vector<int64_t>{total_candidates}, at::dtype<int>());
      // int* output_len_data = output_len->mutable_data<int>();
      // memset(output_len_data, 0, total_candidates * sizeof(int));
      // float* output_prob_data = output_prob->mutable_data<float>();

      // std::vector<int32_t> values_cache;

      // auto it = A_next_inv.begin();
      // for (int index = 0; index < top_paths; index++, it++) {
      //   if (it == A_next_inv.end()) { break; }
      //   auto& candidate = it->second;
      //   output_len_data[b * top_paths + index] = candidate.size();
      //   output_prob_data[b * top_paths + index] = Pb.back()[candidate] + Pnb.back()[candidate];
      //   values_cache.insert(values_cache.end(), candidate.begin(), candidate.end());
      // }
    }
  }
};

}  // namespace

REGISTER_CTC_BEAM_SEARCH_DECODER_KERNELS(DeviceType::kCPU, float);
REGISTER_CTC_BEAM_SEARCH_DECODER_KERNELS(DeviceType::kCPU, double);

}  // namespace oneflow
