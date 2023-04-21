import argparse
import torch
import os
from models.model import EDN
from collections import OrderedDict
from tools.utils import check_size
try:
  import onnx
  import onnxruntime
except ImportError as e:
  raise ImportError(f'Please install onnx and onnxruntime first. {e}')

def verify_onnx_model(model_path):
  onnx_model = onnx.load(model_path)
  #check that the model is well formed
  onnx.checker.check_model(onnx_model)

  #representation of the graph
  # print(onnx.helper.printable_graph(onnx_model.graph))

def convert_to_onnx(model, save_path, gpu = False):
  if 'LiteEX' in args.pretrained:
      # EDN-LiteEX
      input = torch.randn(1, 3, 224, 224)
  else:
      input = torch.randn(1, 3, 384, 384)
  if gpu:
      input = input.cuda()
  print(input.shape)
  with torch.no_grad():
    torch.onnx.export(
      model,
      input,
      save_path,
      export_params=True,
      opset_version=11,#default 13
      do_constant_folding=True, #to execute constant folding for optimization
      verbose=False,
      input_names=["img"],
      output_names=["out"],
      # output_names=["d0"],
      # dynamic_axes={}
    )

def build_model(args):
    if 'Lite' in args.pretrained:
        backbone_arch = 'mobilenetv2'
    elif 'VGG16' in args.pretrained:
        backbone_arch = 'vgg16'
    elif 'R50' in args.pretrained:
        backbone_arch = 'resnet50'
    elif 'P2T-S' in args.pretrained:
        backbone_arch = 'p2t_small'
    else:
        raise NotImplementedError("recognized unknown backbone given the model_path")

    model = EDN(arch=backbone_arch)
    if not os.path.isfile(args.pretrained):
        print('Pre-trained model file does not exist...')
        exit(-1)
    state_dict = torch.load(args.pretrained)
    new_keys = []
    new_values = []
    for key, value in zip(state_dict.keys(), state_dict.values()):
        new_keys.append(key.replace('module.', ''))
        new_values.append(value)
        new_dict = OrderedDict(list(zip(new_keys, new_values)))
    model.load_state_dict(new_dict, strict=False)

    if args.gpu:
        model = model.cuda()
    # set to evaluation mode
    model.eval()

    return model

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--pretrained", type=str, required=True)
  parser.add_argument('--gpu', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU')
  # parser.add_argument("--jit", action="store_true", default="False")
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  net = build_model(args=args)

  onnx_path = f"{os.path.basename(args.pretrained).split('.')[0]}.onnx"
  convert_to_onnx(net, onnx_path, gpu=args.gpu)
  verify_onnx_model(onnx_path)
