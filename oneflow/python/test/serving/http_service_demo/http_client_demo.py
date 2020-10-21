import argparse
import cv2
import http.client
import json
import os
import time

def get_parser():
    parser = argparse.ArgumentParser("flags for http client demo")
    parser.add_argument("--server_address", type = str, default = "localhost", help = "")
    parser.add_argument("--server_port", type = int, default = 8000, help = "")
    parser.add_argument("--test_images_path", type = str, default = "./client_images", help = "")
    return parser

parser = get_parser()
args = parser.parse_args()

conn = http.client.HTTPConnection(args.server_address, args.server_port)
conn.request("GET", "/")
r1 = conn.getresponse()
print(r1.status, r1.reason)
print(r1.read().decode())

headers = {'Content-type': 'application/json'}
images = os.listdir(args.test_images_path)

while True:
    for im in images:
        img = cv2.imread(os.path.join(args.test_images_path, im))
        res = {"image": json.dumps(img.tolist()) }
        json_data = json.dumps(res)

        print("\n##############################")
        print("send image %s to server" % im)
        conn.request('POST', '/post', json_data, headers)
        response = conn.getresponse()
        print("get predicted result from server: %s" % response.read().decode())
        print("##############################\n")

        time.sleep(2)

# conn.close()



