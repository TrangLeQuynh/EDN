MODEL_NAME=EDN-Lite

echo "converting pytorch to onnx......"
python pytorch2onnx.py --pretrained pretrained/${MODEL_NAME}.pth

echo "converting onnx to tf (.pb)......"
python onnx2tf.py --onnx ${MODEL_NAME}.onnx

echo "converting tf (.pb) to tflite......"
python tf2tflite.py --tf ${MODEL_NAME} --tflite ${MODEL_NAME}.tflite

#remove pb and onnx
rm -rf $MODEL_NAME
rm -rf ${MODEL_NAME}.onnx
