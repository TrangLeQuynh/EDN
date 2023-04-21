import argparse
import tensorflow as tf
from tflite_support import metadata_schema_py_generated as _metadata_fb
from tflite_support import metadata as _metadata 
import flatbuffers

"""
TensorFlow FrozenGraph (.pb) -> TensorFlow Lite(.tflite)
"""
def tf2tflite(tf_path, tflite_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

"""
Create the metadata for EDN lite
"""
def create_metadata():
    """Creates model info."""
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "EDN-Lite"
    model_meta.description = (
        "EDN: Extremely Downsampled Network, "
        "which employs an extreme downsampling technique to effectively learn a global view of the whole image, "
        "leading to accurate salient object localization."
        "..."
    )

    """Create input info."""
    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "img"
    # input_meta.description = ("")
    input_normalization_1 = _metadata_fb.ProcessUnitT()
    input_normalization_1.optionsType = (_metadata_fb.ProcessUnitOptions.NormalizationOptions)
    input_normalization_1.options = _metadata_fb.NormalizationOptionsT()
    input_normalization_1.options.std = [255]
    input_normalization_2 = _metadata_fb.ProcessUnitT()
    input_normalization_2.optionsType = (_metadata_fb.ProcessUnitOptions.NormalizationOptions)
    input_normalization_2.options = _metadata_fb.NormalizationOptionsT()
    input_normalization_2.options.mean = [0.406, 0.456, 0.485]
    input_normalization_2.options.std = [0.225, 0.224, 0.229]

    input_meta.processUnits = [input_normalization_1, input_normalization_2]

    """Create output info."""
    output_meta =  _metadata_fb.TensorMetadataT()
    output_meta.name = "out"

    """Create subgraph info."""
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta.subgraphMetadata = [subgraph]

    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b), 
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER
    )
    metadata_buf = b.Output()
    return metadata_buf

def add_metadata(tflite_path, verify=False):
    metadata_buf = create_metadata()
    #populate metadata to the model file
    populator = _metadata.MetadataPopulator.with_model_file(tflite_path)
    populator.load_metadata_buffer(metadata_buf)
    populator.populate()

    if verify:
      verify_metadata(tflite_path)

def verify_metadata(tflite_path):
    displayer = _metadata.MetadataDisplayer.with_model_file(tflite_path)
    json_file = displayer.get_metadata_json()
    print(json_file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", type=str, required=True)
    parser.add_argument("--tflite", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tf2tflite(tf_path=args.tf, tflite_path=args.tflite)
    add_metadata(args.tflite, verify=True)
