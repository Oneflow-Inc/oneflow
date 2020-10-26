import argparse
import cv2
import grpc
import os
import prediction_service_pb2_grpc as grpc_service_pb2
import prediction_service_pb2 as predict_message_pb2
import time

def get_parser():
    parser = argparse.ArgumentParser("flags for http client demo")
    parser.add_argument("--server_address", type = str, default = "localhost", help = "")
    parser.add_argument("--server_port", type = int, default = 8000, help = "")
    parser.add_argument("--test_images_path", type = str, default = "/dataset/http_service_demo_client_images/", help = "")
    return parser

parser = get_parser()
args = parser.parse_args()

if not os.path.exists(args.test_images_path):
    raise Exception('Images path does not existed.')

class PredictionServiceClient(object):
    def __init__(self):
        self.host = args.server_address
        self.server_port = args.server_port
        # instantiate a channel
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port))
        # bind the client and the server
        self.stub = grpc_service_pb2.PredictionServiceStub(self.channel)

    def get_predict_result(self, image):
        request = predict_message_pb2.PredictRequest()
        request.np_array_content = image.tobytes()
        for dim in image.shape:
            request.np_array_shapes.append(dim)
        return self.stub.Predict(request)

client = PredictionServiceClient()

images = os.listdir(args.test_images_path)

while True:
    for im in images:
        img = cv2.imread(os.path.join(args.test_images_path, im))

        print("\n##############################\n")
        print("send image %s to server" % im)
        result = client.get_predict_result(img)
        print(f'{result}')
        print("##############################\n")

        time.sleep(2)
