import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)

func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)

data_dir="/dataset/imagenet_16_same_pics/ofrecord"

@flow.function(func_config)
def DataLoaderJob():
    batch_size = 8
    seed = 0
    rgb_mean = [123.68, 116.779, 103.939]
    rgb_std = [58.393, 57.12, 57.375]
    with flow.fixed_placement("cpu", "0:0"):
        ofrecord = flow.data.ofrecord_loader(data_dir, batch_size=batch_size)
        image = flow.data.OFRecordImageDecoderRandomCrop(ofrecord, "encoded", seed=seed, color_space="RGB")
        label = flow.data.OFRecordRawDecoder(ofrecord, "class/label", shape=(), dtype=flow.int32)
        rsz = flow.image.Resize(image, resize_x=224, resize_y=224, color_space="RGB")
        print(rsz.shape)
        print(label.shape)

        rng = flow.random.CoinFlip(batch_size=batch_size, seed=seed)
        normal = flow.image.CropMirrorNormalize(rsz, mirror_blob=rng, color_space="RGB",
                mean=rgb_mean, std=rgb_std, output_dtype = flow.float)
        print(normal.shape)
        return rsz, normal, label, rng

rsz, normal, label, rng= DataLoaderJob().get()
print("resized image: ", rsz)
print("normalized image output: ", normal)
print("label: ", label)
print("mirror:", rng)
np.save('output/oneflow_train_data_0.npy', normal.ndarray())

rsz, normal, label, rng= DataLoaderJob().get()
print("resized image: ", rsz)
print("normalized image output: ", normal)
print("label: ", label)
print("mirror:", rng)
np.save('output/oneflow_train_data_1.npy', normal.ndarray())

