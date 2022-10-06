"""
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
"""


def BalancedPartNums(total, part_size):
    base = int(total / part_size)
    remainder = total % part_size
    return [base + int(i < remainder) for i in range(part_size)]


def BalancedRanges(total, part_size):
    balanced_part_nums = BalancedPartNums(total, part_size)
    ranges = []
    start = 0
    for part_num in balanced_part_nums:
        end = start + part_num
        ranges.append((start, end))
        start = end
    return ranges
