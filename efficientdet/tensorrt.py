from absl import app, logging, flags
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('saved_model_dir', default='./test_saved_model',
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')

def input_fn(batch_size, height, width):
    features = np.random.normal(
      loc=112, scale=70,
      size=(batch_size, height, width, 3)).astype(np.float32)
    features = np.clip(features, 0.0, 255.0).astype(tf.float32)
    features = tf.convert_to_tensor(value=tf.compat.v1.get_variable(
      "features", initializer=tf.constant(features)))
    dataset = tf.data.Dataset.from_tensor_slices([features])
    dataset = dataset.take(1)
    for image in dataset:
        yield (image,)


def main(_):
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode='FP16')
    converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=FLAGS.saved_model_dir, conversion_params=params)
    converter.convert()

    converter.build(input_fn=lambda: input_fn(1, 512, 512))
    tf.io.gfile.rmtree(FLAGS.saved_model_dir)
    converter.save(FLAGS.saved_model_dir)
    logging.info('Tensorrt model saved at ' + FLAGS.saved_model_dir)

if __name__=='__main__':
    app.run(main)