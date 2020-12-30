import tensorflow as tf
import sys; sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import loss_functions as lf
from tensorflow.keras import backend as K
tf.enable_eager_execution()
tfe = tf.contrib.eager

# If you select `use_tanh=True' then predictions are in limit [-1, 1] and so loss and gradients
# return numbers. If you don't, predictions are well outside range [-1, 1] (by construction)
# and so you will get a loss that in `inf' and a gradient w.r.t. weight = `nan',
# a gradient w.r.t. bias = `inf'.


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.W = tfe.Variable(5., name='weight')
        self.B = tfe.Variable(10., name='bias')

    def call(self, inputs):
        if use_tanh is True:
            return tf.atan(inputs * self.W + self.B)
        else:
            return inputs * self.W + self.B


def main():
    n = 10
    x = tf.random_normal([n, 2])
    noise = tf.random_normal([n, 2])
    y = tf.atan(x * 3 + 2) + noise

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    l = lf.cauchy_selection_loss_fixed_boundary(gamma=0.2)

    model = Model()
    with tf.GradientTape() as tape:
        y_pred = model(x)
        print(y_pred)
        loss_value = tf.reduce_mean(l(y, model(x)))

    gradients = tape.gradient(loss_value, model.variables)
    print("Gradient of the weight is " + str(K.get_value(gradients[0])))
    print("Gradient of the bias is " + str(K.get_value(gradients[1])))

    optimizer.apply_gradients(zip(gradients, model.variables),
                              global_step=tf.train.get_or_create_global_step())
    print("Updated weights and biases are: ")
    print(model.variables[0])
    print(model.variables[1])

if __name__ == "__main__":
    use_tanh = True
    main()
    # Gradient of the weight is nan
    # Gradient of the bias is inf
    # Updated weights and biases are:
    # <tf.Variable 'weight:0' shape=() dtype=float32, numpy=nan>
    # <tf.Variable 'bias:0' shape=() dtype=float32, numpy=-inf>

    # The predictions y = model(x) will now all be nan


