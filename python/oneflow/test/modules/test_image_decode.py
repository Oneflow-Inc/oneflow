import unittest
import numpy as np
import cv2
import oneflow as flow

@flow.unittest.skip_unless_1n1d()
class TestImageDecode(flow.unittest.TestCase):

    def test_image_decode(test_case):
        images = ['/dataset/mscoco_2017/val2017/000000000139.jpg', '/dataset/mscoco_2017/val2017/000000000632.jpg']
        image_files = [open(im, 'rb') for im in images]
        images_bytes = [imf.read() for imf in image_files]
        static_shape = (len(images_bytes), max([len(bys) for bys in images_bytes]))
        for imf in image_files:
            imf.close()
        image_decoder = flow.nn.image.decode(color_space='BGR')
        images_np_arr = [np.frombuffer(bys, dtype=np.byte).reshape(1, -1) for bys in images_bytes]
        images_np_arr_static = np.zeros(static_shape, dtype=np.int8)
        for (idx, np_arr) in enumerate(images_np_arr):
            images_np_arr_static[idx, :np_arr.shape[1]] = np_arr
        input = flow.Tensor(images_np_arr_static, dtype=flow.int8, device=flow.device('cpu'))
        images_buffer = flow.tensor_to_tensor_buffer(input, instance_dims=1)
        decoded_images_buffer = image_decoder(images_buffer)
        of_decoded_images = decoded_images_buffer.numpy()
        cv2_images = [cv2.imread(image) for image in images]
        cv2_decoded_images = [np.array(image) for image in cv2_images]
        for (of_decoded_image, cv2_decoded_image) in zip(of_decoded_images, cv2_decoded_images):
            test_case.assertTrue(len(of_decoded_image.shape) == 3)
            test_case.assertTrue(len(cv2_decoded_image.shape) == 3)
            test_case.assertTrue(np.allclose(of_decoded_image, cv2_decoded_image))
if __name__ == '__main__':
    unittest.main()