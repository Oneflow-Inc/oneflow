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
import numpy as np
import oneflow as flow

# flow.enable_eager_execution(True)
flow.config.gpu_device_num(1)

func_config = flow.FunctionConfig()
func_config.default_logical_view(flow.scope.consistent_view())
func_config.default_data_type(flow.float)

data_dir = "/dataset/imagenet_16_same_pics/ofrecord"


@flow.global_function(func_config)
def DataLoaderJob():
    batch_size = 8
    seed = 0
    rgb_mean = [123.68, 116.779, 103.939]
    rgb_std = [58.393, 57.12, 57.375]

    # ofrecord = flow.data.ofrecord_loader(data_dir, batch_size=batch_size)
    ofrecord = flow.data.ofrecord_reader(
        data_dir, batch_size=batch_size, random_shuffle=True
    )
    image = flow.data.OFRecordImageDecoderRandomCrop(
        ofrecord,
        "encoded",
        seed=seed,
        color_space="RGB",
        name="of_record_iamge_decoder_random_crop",
    )
    label = flow.data.OFRecordRawDecoder(
        ofrecord, "class/label", shape=(), dtype=flow.int32
    )
    rsz = flow.image.Resize(image, resize_x=224, resize_y=224, color_space="RGB")
    print(rsz.shape)
    print(label.shape)

    rng = flow.random.CoinFlip(batch_size=batch_size, seed=seed, name="coin_flip")
    normal = flow.image.CropMirrorNormalize(
        rsz,
        mirror_blob=rng,
        color_space="RGB",
        mean=rgb_mean,
        std=rgb_std,
        output_dtype=flow.float,
    )
    print(normal.shape)
    return rsz, normal, label, rng


@flow.global_function(func_config)
def DataLoaderEvalJob():
    batch_size = 8
    rgb_mean = [123.68, 116.779, 103.939]
    rgb_std = [58.393, 57.12, 57.375]

    # ofrecord = flow.data.ofrecord_loader(
    #     data_dir,
    #     batch_size=batch_size,
    #     part_name_suffix_length=5,
    #     data_part_num=1,
    #     shuffle=False,
    # )
    ofrecord = flow.data.ofrecord_reader(
        data_dir,
        batch_size=batch_size,
        part_name_suffix_length=5,
        data_part_num=1,
        random_shuffle=False,
    )
    image = flow.data.OFRecordImageDecoder(ofrecord, "encoded", color_space="RGB")
    label = flow.data.OFRecordRawDecoder(
        ofrecord, "class/label", shape=(), dtype=flow.int32
    )
    rsz = flow.image.Resize(image, resize_shorter=256, color_space="RGB")

    normal = flow.image.CropMirrorNormalize(
        rsz,
        color_space="RGB",
        crop_h=224,
        crop_w=224,
        crop_pos_y=0.5,
        crop_pos_x=0.5,
        mean=rgb_mean,
        std=rgb_std,
        output_dtype=flow.float,
    )
    return normal, label


if __name__ == "__main__":
    rsz, normal, label, rng = DataLoaderJob().get()
    print("resized image: ", rsz)
    print("normalized image output: ", normal)
    print("label: ", label)
    print("mirror:", rng)
    np.save("output/oneflow_train_data_0.npy", normal.numpy())

    rsz, normal, label, rng = DataLoaderJob().get()
    print("resized image: ", rsz)
    print("normalized image output: ", normal)
    print("label: ", label)
    print("mirror:", rng)
    np.save("output/oneflow_train_data_1.npy", normal.numpy())

    normal, label = DataLoaderEvalJob().get()
    print("normalized image output: ", normal)
    print("label: ", label)
    np.save("output/oneflow_eval_data_0.npy", normal.numpy())
