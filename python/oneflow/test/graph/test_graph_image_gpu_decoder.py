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
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest


class OFRecordDataLoader(flow.nn.Module):
    def __init__(self):
        super().__init__()
        batch_size = 4
        image_size = 224
        self.train_record_reader = flow.nn.OFRecordReader(
            flow.unittest.dataset_dir("imagenette/ofrecord"),
            batch_size=batch_size,
            data_part_num=1,
            part_name_suffix_length=5,
            random_shuffle=True,
            shuffle_after_epoch=True,
            # placement=flow.placement("cpu", ranks=[0]),
            # sbp=[flow.sbp.broadcast]
        )

        self.record_label_decoder = flow.nn.OFRecordRawDecoder(
            "class/label", shape=(), dtype=flow.int32
        )
        self.bytes_decoder = flow.nn.OFRecordBytesDecoder("encoded")
        self.image_gpu_decoder = flow.nn.OFRecordImageGpuDecoderRandomCropResize(
            target_width=image_size, target_height=image_size, num_workers=3
        )

        color_space = "RGB"
        output_layout = "NHWC"

        self.flip = flow.nn.CoinFlip(
            batch_size=batch_size,
            # placement=flow.placement("cpu", ranks=[0]),
            # sbp=[flow.sbp.broadcast]
        )

        rgb_mean = [123.68, 116.779, 103.939]
        rgb_std = [58.393, 57.12, 57.375]
        self.crop_mirror_norm = flow.nn.CropMirrorNormalize(
            color_space=color_space,
            output_layout=output_layout,
            mean=rgb_mean,
            std=rgb_std,
            output_dtype=flow.float,
        )

    def forward(self) -> (flow.Tensor, flow.Tensor):
        train_record = self.train_record_reader()
        label = self.record_label_decoder(train_record)
        encoded = self.bytes_decoder(train_record)
        image = self.image_gpu_decoder(encoded)
        rng = self.flip()
        if image.is_cuda:
            rng = rng.to("cuda")
        image = self.crop_mirror_norm(image, rng)
        return image, label


_DECODE_SCRIPT = """
import sys
import cv2
import numpy as np
import oneflow as flow

root, size = sys.argv[1], int(sys.argv[2])


class Loader(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.reader = flow.nn.OFRecordReader(
            root, batch_size=4, data_part_num=1, part_name_suffix_length=5, random_shuffle=False
        )
        self.label = flow.nn.OFRecordRawDecoder("class/label", shape=(), dtype=flow.int32)
        self.encoded = flow.nn.OFRecordBytesDecoder("encoded")
        self.decoder = flow.nn.OFRecordImageGpuDecoderRandomCropResize(
            target_width=size,
            target_height=size,
            num_workers=2,
            random_area=[1.0, 1.0],
            random_aspect_ratio=[1.0, 1.0],
        )

    def forward(self):
        record = self.reader()
        return self.decoder(self.encoded(record)), self.label(record)


class G(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.loader = Loader()

    def build(self):
        return self.loader()


images = np.load(root + "/images.npy", allow_pickle=True)
image, label = G()()
assert image.is_cuda
worst = 0.0
for out, k in zip(image.numpy(), label.numpy()):
    ref = cv2.cvtColor(cv2.imdecode(images[k], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ref = cv2.resize(ref, (size, size), interpolation=cv2.INTER_LINEAR)
    worst = max(worst, np.abs(out.astype(np.int32) - ref.astype(np.int32)).mean())
print("worst mean abs diff", worst, flush=True)
assert worst < 3.0
"""


def _write_jpeg_ofrecord(root):
    import cv2
    from oneflow.core.record import record_pb2

    rng = np.random.RandomState(0)
    images = []
    # Square images, so the random crop (area 1, aspect ratio 1) keeps the whole image.
    for i, (h, w) in enumerate([(224, 224), (97, 97), (301, 301), (256, 256)]):
        image = cv2.GaussianBlur((rng.rand(h, w, 3) * 255).astype(np.uint8), (9, 9), 4)
        flags = [cv2.IMWRITE_JPEG_QUALITY, 92]
        if i == 3:
            flags += [cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
        ok, buf = cv2.imencode(".jpg", image, flags)
        assert ok
        images.append(buf)
    np.save(os.path.join(root, "images.npy"), np.array(images, dtype=object), allow_pickle=True)
    with open(os.path.join(root, "part-00000"), "wb") as f:
        for k in range(len(images)):
            record = record_pb2.OFRecord()
            record.feature["encoded"].bytes_list.value.append(images[k].tobytes())
            record.feature["class/label"].int32_list.value.append(k)
            data = record.SerializeToString()
            f.write(struct.pack("<q", len(data)))
            f.write(data)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestImageGpuDecoderGraph(oneflow.unittest.TestCase):
    @unittest.skip("skip for now, becase it failed 2 times in past week")
    def test_image_gpu_decoder_graph(test_case):
        cc_reader = OFRecordDataLoader()

        class GraphReader(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.my_reader = cc_reader

            def build(self):
                return self.my_reader()

        reader_g = GraphReader()
        image, label = reader_g()
        print(image.shape)
        print(label)

    # Without a usable hardware decoder (opted out, or one that cannot be created, as with no
    # VA-API driver) every image must still decode, on the CPU.
    def test_image_gpu_decoder_matches_opencv(test_case):
        with tempfile.TemporaryDirectory() as root:
            _write_jpeg_ofrecord(root)
            for extra_env in [
                {},
                {"ONEFLOW_DECODER_ENABLE_NVJPEG_HARDWARE_ACCELERATION": "0"},
                {"LIBVA_DRIVER_NAME": "oneflow_test_no_such_driver"},
            ]:
                proc = subprocess.run(
                    [sys.executable, "-c", _DECODE_SCRIPT, root, "128"],
                    env=dict(os.environ, **extra_env),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    timeout=600,
                )
                test_case.assertEqual(proc.returncode, 0, str(extra_env) + proc.stdout)


if __name__ == "__main__":
    unittest.main()
