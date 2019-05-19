import tensorflow as tf

a = tf.random.normal([4,28,28,3])

print(a.shape,a.ndim)

a = tf.reshape(a,[4,784,3])
print(a.shape)