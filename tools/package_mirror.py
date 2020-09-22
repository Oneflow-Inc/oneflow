import glob
import argparse
import os
import re

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


# oss://oneflow-static/...
def convert_url_to_oss_url():
    raise NotImplemented


# https://oneflow-static.oss-cn-beijing.aliyuncs.com/...
def convert_url_to_oss_https_url(url):
    raise NotImplemented


def upload_to_aliyun(dir_path):
    urls = scan_urls(dir_path)
    for url in urls:
        oss_url = convert_url_to_oss_url(url)
    raise NotImplemented


if __name__ == "__main__":
    if args.src_path != None:
        upload_to_aliyun(args.src_path)
    if args.url != None:
        oss_url = convert_url_to_oss_https(args.url)
        print(oss_url)
