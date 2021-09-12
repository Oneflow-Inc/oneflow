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

#include <string>
#include <stdlib.h>
#include "oneflow/api/cpp/inference_session.h"

using namespace oneflow;

char* load_image(std::string image_path) {
    // TODO
    return nullptr;
}

int main() {
    float* image_mock_data = new float[28*28];
    for(int i = 0; i < 28*28; i++) {
        image_mock_data[i] = std::rand() / double(RAND_MAX);
    }
    std::string job_name = "mlp_inference";
    SessionOption option;
    option.device_tag = "cpu";
    option.ctrl_port = 11235;
    option.device_num = 1;

    InferenceSession session(option);
    std::string model_path = "/home/allen/projects/oneflow-serving/lenet_models/";
    ModelVersionPolicy ver_policy;
    ver_policy.latest = true;
    ver_policy.version = 1;
    std::string basename = "saved_model";
    session.LoadModel(model_path, ver_policy, basename);
    session.Launch();

    std::map<std::string, std::shared_ptr<Tensor>> tensor_inputs, tensor_outputs;
    std::string image_path = "/home/allen/projects/oneflow-serving/7.png";
    //char* image_data = load_image(image_path);
    tensor_inputs["Input_21"] = Tensor::fromBlob((char*)image_mock_data, {1,1,28,28}, kFloat);
    tensor_outputs = session.Run(job_name, tensor_inputs);

     for (auto& entry : tensor_outputs) {
        std::cout << "Output name:" << entry.first << std::endl;
        //TODO
    }
    session.Close();

    return 0;
}
