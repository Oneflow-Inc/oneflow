import os
import numpy as np
import oneflow as of

blob_watcher = None
blob_watched = {}
diff_blob_watched = {}


def save_blob_watched(iter):
    saver_path = os.path.join("saver", "iter" + str(iter), "fw")
    diff_saver_path = os.path.join("saver", "iter" + str(iter), "bw")

    if not os.path.exists(saver_path):
        os.makedirs(saver_path)

    if not os.path.exists(diff_saver_path):
        os.makedirs(diff_saver_path)

    global blob_watched
    for lbn, blob_data in blob_watched.items():
        if "blob_def" in blob_data:
            blob_def = blob_data["blob_def"]
            op_saver_path = os.path.join(saver_path, blob_def.op_name)
            if not os.path.exists(op_saver_path):
                os.makedirs(op_saver_path)
            np.save(os.path.join(op_saver_path, blob_def.blob_name), blob_data["blob"].ndarray())
        else:
            print("no blob_def found for: {}".format(lbn))

    global diff_blob_watched
    for lbn, blob_data in diff_blob_watched.items():
        if "blob_def" in blob_data:
            blob_def = blob_data["blob_def"]
            op_saver_path = os.path.join(diff_saver_path, blob_def.op_name)
            if not os.path.exists(op_saver_path):
                os.makedirs(op_saver_path)
            np.save(os.path.join(op_saver_path, blob_def.blob_name), blob_data["blob"].ndarray())
        else:
            print("no blob_def found for diff of: {}".format(lbn))


class BlobWatcher(object):
    def __init__(self, start_iter=0, base_dir="dump_blobs"):
        self.cur_iter = start_iter
        self.base_dir = base_dir

        save_dir = os.path.join(self.base_dir, "iter{}".format(self.cur_iter))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def step(self):
        self.cur_iter += 1

        save_dir = os.path.join(self.base_dir, "iter{}".format(self.cur_iter))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def watch(self, blob, watch_diff=False):
        saver = None
        if hasattr(blob, "sub_consistent_blob_list"):
            save_path_list = [
                self.get_blob_grad_save_path(sub_blob) for sub_blob in blob.sub_consistent_blob_list
            ]

            def make_save_fn(save_path):
                def save_fn(x):
                    self.save(save_path, x.ndarray())

                return save_fn

            saver = [make_save_fn(save_path) for save_path in save_path_list]
        else:
            save_path = self.get_blob_save_path(blob)

            def save_fn(x):
                self.save(save_path, x.ndarray())

            saver = save_fn

        of.watch(blob, saver)
        if watch_diff:
            self.watch_diff(blob)

    def watch_diff(self, blob):
        saver = None
        if hasattr(blob, "sub_consistent_blob_list"):
            save_path_list = [
                self.get_blob_grad_save_path(sub_blob) for sub_blob in blob.sub_consistent_blob_list
            ]

            def make_save_fn(save_path):
                def save_fn(x):
                    self.save(save_path, x.ndarray())

                return save_fn

            saver = [make_save_fn(save_path) for save_path in save_path_list]
        else:
            save_path = self.get_blob_grad_save_path(blob)

            def save_fn(x):
                self.save(save_path, x.ndarray())

            saver = save_fn

        of.watch_diff(blob, saver)

    def get_blob_save_path(self, blob):
        return os.path.join(
            self.base_dir,
            "iter{}".format(self.cur_iter),
            "{}-{}".format(blob.op_name, blob.blob_name),
        )

    def get_blob_grad_save_path(self, blob):
        return os.path.join(
            self.base_dir,
            "iter{}".format(self.cur_iter),
            "{}-{}_grad".format(blob.op_name, blob.blob_name),
        )

    def save(self, path, ndarray):
        np.save(path, ndarray)


def get_blob_watcher():
    global blob_watcher
    if blob_watcher is None:
        blob_watcher = BlobWatcher()
    return blob_watcher
