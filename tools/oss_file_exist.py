import os
import oss2


def check_existence(endpoint, bucket, path):
    ki = os.getenv("OSS_ACCESS_KEY_ID")
    ks = os.getenv("OSS_ACCESS_KEY_SECRET")
    auth = oss2.Auth(ki, ks)
    bucket_obj = oss2.Bucket(auth, endpoint, bucket)
    files = bucket_obj.list_objects(path)
    file_cnt = 0
    for f in files.object_list:
        file_cnt += 1
    is_existed = bucket_obj.object_exists(path) or file_cnt > 0
    if is_existed:
        print("export OSS_FILE_EXISTED=1")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        required=False,
        default="oss-cn-beijing.aliyuncs.com",
    )
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)

    args = parser.parse_args()

    check_existence(args.endpoint, args.bucket, args.path)
