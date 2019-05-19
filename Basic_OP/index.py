import tensorflow as tf



a = tf.range(10)
print(a)

b = tf.gather(a,axis=0,indices=[2,3])
print(b)