import glob
import argparse
import os
import re
import oss2
from urllib.parse import urlparse
import requests

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
            urls += re.findall(r'https?://[^\s<>"\)]+|www\.[^\s<>"]+', content)
    return urls


def convert_url_to_oss_key(url):
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert not parsed.params
    assert not parsed.query
    assert not parsed.port
    assert not parsed.fragment
    assert parsed.path.startswith("/")
    path = parsed.path[1:-1]
    return os.path.join("third_party_mirror", parsed.scheme, parsed.netloc, path)


# https://oneflow-static.oss-cn-beijing.aliyuncs.com/...
def convert_url_to_oss_https_url(url):
    raise NotImplemented


def download_file(url):
    path = None
    return path


def is_mirrored(url):
    return False


def already_exists(url):
    return False


def should_be_mirrored(url: str):
    parsed = urlparse(url)
    return (
        not parsed.port
        and not parsed.query
        and not parsed.params
        and url.endswith(("gz", "tar", "zip"))
        and already_exists(url) == False
    )


def upload_one_to_aliyun(url):
    ki = os.getenv("OSS_ACCESS_KEY_ID")
    ks = os.getenv("OSS_ACCESS_KEY_SECRET")
    auth = oss2.Auth(ki, ks)
    endpoint = "oss-cn-beijing.aliyuncs.com"
    bucket = oss2.Bucket(auth, endpoint, "oneflow-static")
    key = convert_url_to_oss_key(url)
    bucket.put_object(key, requests.get(url).raw.read())


def upload_to_aliyun(dir_path):
    urls = scan_urls(dir_path)
    for url in urls:
        if should_be_mirrored(url):
            print("mirroring: ", url)
            upload_one_to_aliyun(url)
        else:
            print("skipped: ", url)
            continue


if __name__ == "__main__":
    if args.src_path != None:
        upload_to_aliyun(args.src_path)
    if args.url != None:
        oss_url = convert_url_to_oss_https(args.url)
        print(oss_url)
