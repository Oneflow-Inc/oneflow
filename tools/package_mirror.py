import glob
import argparse
import os
import re
import oss2
from urllib.parse import urlparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--src_path", type=str, required=False)
parser.add_argument("-u", "--url", type=str, required=False)
args = parser.parse_args()


def glob_by_pattern(dir_path, pattern):
    result = []
    for x in glob.glob(os.path.join(dir_path, pattern), recursive=True):
        result.append(x)
    return result


def scan_urls(dir_path):
    cmakes = glob_by_pattern(dir_path, "**/*.cmake")
    urls = []
    for cmake_path in cmakes:
        with open(cmake_path) as f:
            content = f.read()
            urls += re.findall("https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", content)
    return urls


def convert_url_to_oss_key(url):
    parsed = urlparse(url)
    print(parsed.port)
    # parsed.scheme = ""
    return parsed.geturl()


# oss://oneflow-static/...
def convert_url_to_oss_url(url):
    key = convert_url_to_oss_key(url)
    return os.path.join("oss://oneflow-static", key)


# https://oneflow-static.oss-cn-beijing.aliyuncs.com/...
def convert_url_to_oss_https_url(url):
    raise NotImplemented


def download_file(url):
    path = None
    return path


def is_mirrored(url):
    return False


def upload_one_to_aliyun(file_path, oss_url):
    ki = os.getenv("OSS_ACCESS_KEY_ID")
    ks = os.getenv("OSS_ACCESS_KEY_SECRET")
    auth = oss2.Auth(ki, ks)
    endpoint = "oss-cn-beijing.aliyuncs.com"
    bucket = oss2.Bucket(auth, endpoint, "oneflow-static")


def upload_to_aliyun(dir_path):
    urls = scan_urls(dir_path)
    for url in urls:
        if is_mirrored(url):
            print("already mirrored: ", url)
            continue
        else:
            print("to be mirrored: ", url)
            file_path = download_file(url)
            oss_url = convert_url_to_oss_url(url)
            upload_one_to_aliyun(file_path, oss_url)


if __name__ == "__main__":
    if args.src_path != None:
        upload_to_aliyun(args.src_path)
    if args.url != None:
        oss_url = convert_url_to_oss_https(args.url)
        print(oss_url)
