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
        self.blobs_watched = {}

    def save(self, iter):
        save_dir = os.path.join(self.base_dir, "iter{}".format(iter))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for blob_key, blob_ndarray in self.blobs_watched.items():
            if isinstance(blob_ndarray, np.ndarray):
                save_path = os.path.join(save_dir, blob_key)
                np.save(save_path, blob_ndarray)
            else:
                raise ValueError

        self.blobs_watched = {}

    def dump(self, blob_key, blob_ndarray):
        self.blobs_watched[blob_key] = blob_ndarray

    def watch(self, blob, watch_diff=False):
        watcher = None
        if hasattr(blob, "sub_consistent_blob_list"):

            def make_watch_fn(key):
                def watch_fn(x):
                    self.dump(key, x.ndarray())

                return watch_fn

            watcher = [
                make_watch_fn("{}-{}".format(sub_blob.op_name, sub_blob.blob_name))
                for sub_blob in blob.sub_consistent_blob_list
            ]
        else:

            def watch_fn(x):
                self.dump("{}-{}".format(blob.op_name, blob.blob_name), x.ndarray())

            watcher = watch_fn

        of.watch(blob, watcher)
        if watch_diff:
            self.watch_diff(blob)

    def watch_diff(self, blob):
        watcher = None
        if hasattr(blob, "sub_consistent_blob_list"):

            def make_watch_fn(key):
                def watch_fn(x):
                    self.dump(key, x.ndarray())

                return watch_fn

            watcher = [
                make_watch_fn("{}-{}_grad".format(sub_blob.op_name, sub_blob.blob_name))
                for sub_blob in blob.sub_consistent_blob_list
            ]
        else:

            def watch_fn(x):
                self.dump("{}-{}_grad".format(blob.op_name, blob.blob_name), x.ndarray())

            watcher = watch_fn

        of.watch_diff(blob, watcher)


def get_blob_watcher():
    global blob_watcher
    if blob_watcher is None:
        blob_watcher = BlobWatcher()
    return blob_watcher
