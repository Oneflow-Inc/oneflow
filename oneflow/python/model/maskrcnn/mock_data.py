import oneflow as flow
import pickle as pk


class MockData(object):
    def __init__(self, data_file, max_objs_per_img):
        with open(data_file, "rb") as f:
            self._data = pk.load(f)
        self._max_objs_per_img = max_objs_per_img

    def blob_def(self, blob_name):
        if blob_name == "images":
            return flow.input_blob_def(
                shape=self._data["images"].shape, dtype=flow.float32, is_dynamic=True
            )
        elif blob_name == "image_size":
            return flow.input_blob_def(
                shape=(len(self._data["image_size"]), 2), dtype=flow.int32
            )
        elif blob_name == "gt_bbox":
            return flow.input_blob_def(
                shape=(len(self._data["gt_bbox"]), self._max_objs_per_img, 4),
                dtype=flow.float32,
                num_of_lod_levels=2,
            )
        elif blob_name == "gt_labels":
            return flow.input_blob_def(
                shape=(len(self._data["gt_labels"]), self._max_objs_per_img),
                dtype=flow.int32,
                num_of_lod_levels=2,
            )
        elif blob_name == "gt_segm":
            segm_mask_shape = [
                len(self._data["gt_segm"]),
                self._max_objs_per_img,
            ] + list(self._data["gt_segm"][0].shape[1:])
            return flow.input_blob_def(
                shape=tuple(segm_mask_shape),
                dtype=flow.int8,
                num_of_lod_levels=2,
            )
        else:
            raise ValueError("Blob is nonexistent")

    def blob(self, blob_name):
        return self._data[blob_name]
