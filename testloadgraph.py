import argparse
import tensorflow as tf
from PIL import Image
import numpy as np

def load_graph(fz_gh_fn):
    with tf.gfile.GFile(fz_gh_fn,"rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map = None,
                return_elements = None,
                name = "prefix"
            )
    return graph

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--fz_model_fn",default = "./emotion_model_frozen.pb",type=str,help="Frozen model file to import")
    args = parser.parse_args()
    graph = load_graph(args.fz_model_fn)

    for op in graph.get_operations():
        print(op.name,op.values())

    x = graph.get_tensor_by_name('prefix/inputs:0')
    y = graph.get_tensor_by_name('prefix/output_node:0')

    img = Image.open('./test.jpg')
    flatten_img = np.asarray(img).reshape(-1, 2304) * 1 / 255.0

    with tf.Session(graph=graph) as sess:
        y_out=sess.run(y,feed_dict={x:flatten_img})
        print(y_out)
    print("finish")
