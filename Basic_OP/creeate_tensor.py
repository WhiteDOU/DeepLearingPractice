import numpy as np

import tensorflow as tf


a = np.arange(5)
print(a)

aa = tf.convert_to_tensor(a,dtype=tf.int32)
print(a.dtype)
print(a.shape)

aa = tf.cast(aa,tf.float32)
print(aa.dtype)