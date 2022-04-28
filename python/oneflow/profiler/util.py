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

NS_IN_SECOND = 1000.0 * 1000.0 * 1000.0
NS_IN_MS = 1000.0 * 1000.0
NS_IN_US = 1000.0

def format_time(time_ns):    
    if time_ns >= NS_IN_SECOND:
        return "{:.3f}s".format(time_ns / NS_IN_SECOND)
    if time_ns >= NS_IN_MS:
        return "{:.3f}ms".format(time_ns / NS_IN_MS)
    if time_ns >= NS_IN_US:
        return "{:.3f}us".format(time_ns / NS_IN_US)
    return "{:.3f}us".format(time_ns)
