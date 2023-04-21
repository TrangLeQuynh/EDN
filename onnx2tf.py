import argparse
import onnx
import torch
import os
from onnx_tf.backend import prepare

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)

    return parser.parse_args()

def convert_onnx2tf(onnx_path, tf_path):
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    #export the model to a .pb file
    tf_rep.export_graph(tf_path)

if __name__ == "__main__":
    args = parse_args()
    model_name = os.path.basename(args.onnx).split('.')[0]

    convert_onnx2tf(onnx_path=args.onnx, tf_path=model_name)
