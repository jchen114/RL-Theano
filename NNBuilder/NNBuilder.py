from _curses import nonl

import numpy as np
import math
import matplotlib.pyplot as plt
import theano
import lasagne
import theano.tensor as T


# Generate data from crazy function
from theano.gof.opt import optimizer


class NN_Build:

    def __init__(self, input_dimension, hidden_units, no_output_units, gamma=0.9, regularization_factor=None):

        self.x = T.dmatrix('x')
        self.y = T.dmatrix('y')

        self.gamma = gamma

        input_layer = lasagne.layers.InputLayer(
            shape=(None, input_dimension),
            input_var=self.x,
            name="Input Layer"
        )

        out_layer = input_layer

        for hidden_unit in hidden_units:
            dense_layer = lasagne.layers.DenseLayer(out_layer, hidden_unit, nonlinearity=lasagne.nonlinearities.rectify)
            out_layer = dense_layer

        self.nn_ftrain = dict()
        self.nn_fpredict = dict()
        self.n_out = dict()

        for output in range(no_output_units):

            self.n_out[output] = (lasagne.layers.DenseLayer(out_layer, num_units=1, nonlinearity=lasagne.nonlinearities.linear))
            params = lasagne.layers.get_all_params(self.n_out[output], trainable=True)
            cost = T.mean(0.5 * lasagne.objectives.squared_error(self.y, lasagne.layers.get_output(self.n_out[output])))
            updates = lasagne.updates.rmsprop(cost, params, 0.001, 0.95, 0.001)

            self.nn_ftrain[output] = (theano.function(
                inputs=[self.x, self.y],
                outputs=[lasagne.layers.get_output(self.n_out[output]), cost],
                updates=updates
            ))

            self.nn_fpredict[output] = theano.function(
                inputs=[self.x],
                outputs=[lasagne.layers.get_output(self.n_out[output])]
            )

    def predict(self, query):
        Q_values = list()
        for a, predictf in self.nn_fpredict.iteritems():
            Q_value = predictf([query])
            Q_values.append(Q_value)
        return Q_values

    def train(self, batch):
        actions = dict()
        for (s, a, r, s_next) in batch:
            if a not in actions:
                actions[a] = list()
            actions[a].append((s,a,r,s_next))

        for a, experiences in actions.iteritems():
            train = (list(), list())
            for (s,a,r,s_next) in experiences:
                predicted_next = self.predict(s_next)  # Q(s',a)
                max_value = np.amax(predicted_next)  # max_a' Q(s', a')
                target = r + self.gamma * max_value
                train[0].append(s)
                train[1].append(target)
            train_x = train[0]
            train_y = np.array(train[1]).reshape((-1,1))
            self.nn_ftrain[a](train_x, train_y)  # x = s, y = r + g * max_a' Q(s', a')


if __name__ == "__main__":
    NN = NN_Build(6, [100,100,100], 4)
    tuples = [ # (s, a, r, s')
        ((1, 1, 1, 1, 1, 1), 1, 1, (1, 1, 1, 1, 1, 2)),
        ((1, 1, 1, 1, 1, 2), 2, 1, (1, 1, 1, 1, 1, 3)),
        ((1, 1, 1, 1, 1, 3), 1, 1, (1, 1, 1, 1, 1, 4)),
        ((1, 1, 1, 1, 1, 4), 2, 1, (1, 1, 1, 1, 1, 5)),
        ((1, 1, 1, 1, 1, 5), 1, 1, (1, 1, 1, 1, 1, 6)),
        ((1, 1, 1, 1, 1, 6), 2, 1, (1, 1, 1, 1, 1, 1))
    ]
    NN.train(tuples)