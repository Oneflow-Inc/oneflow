import os
import numpy as np

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
            np.save(
                os.path.join(op_saver_path, blob_def.blob_name),
                blob_data["blob"].ndarray(),
            )
        else:
            print("no blob_def found for: {}".format(lbn))

    global diff_blob_watched
    for lbn, blob_data in diff_blob_watched.items():
        if "blob_def" in blob_data:
            blob_def = blob_data["blob_def"]
            op_saver_path = os.path.join(diff_saver_path, blob_def.op_name)
            if not os.path.exists(op_saver_path):
                os.makedirs(op_saver_path)
            np.save(
                os.path.join(op_saver_path, blob_def.blob_name),
                blob_data["blob"].ndarray(),
            )
        else:
            print("no blob_def found for diff of: {}".format(lbn))
