import os
from spektral.layers.convolutional.gcn_conv import GCNConv

import tensorflow as tf

from spektral.layers import CrystalConv, GraphSageConv, MessagePassing
from spektral.layers.pooling.global_pool import GlobalMaxPool, GlobalAvgPool, GlobalSumPool

from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.activations import tanh
from tensorflow.sparse import SparseTensor
from tensorflow.keras.backend import epsilon


hidden_states = 64
activation = LeakyReLU(alpha = 0.15)
eps = 1e-5

# Probably needs regularization, but first step is just to fit, then we will regularize.


# Normalize params
normalize5 ={"translate": tf.constant([0, 0, -200, 10000, 0], dtype = tf.float32),
             "scale":     tf.constant([100, 100, 100, 2500, 0.25], dtype = tf.float32),
             "x_dom":  (0, 100),
             "y_dom":  (0, 100),
             "z_dom":  (-200, 100),
             "time":   (10000, 2500),
             "charge": (0, 0.25)}

normalize6 ={"translate": tf.constant([0, 0, -200, 10000, 0, 0], dtype = tf.float32),
             "scale":     tf.constant([100, 100, 100, 2500, 0.25, 1.], dtype = tf.float32),
             "x_dom":  (0, 100),
             "y_dom":  (0, 100),
             "z_dom":  (-200, 100),
             "time":   (10000, 2500),
             "charge": (0, 0.25)}



class GraphSage_network_angles(Model):
    def __init__(self, n_out = 2, n_kappa = 1, n_corr = 0, n_in = 5, hidden_states = 64, forward = False, dropout = 0, gamma = 0, cossin = False, **kwargs):
        super().__init__()
        self.forward = forward
        self.norm_trans   = normalize6['translate'][:n_in]
        self.norm_scale   = normalize6['scale'][:n_in]
        self.gamma        = gamma
        self.n_corr       = n_corr
        self.n_sigs       = n_kappa
        self.cossin       = cossin

        self.batch_edge  = BatchNormalization()

        self.MP          = MP(hidden_states, hidden_states, dropout = dropout)

        self.GraphSage1  = GraphSageConv(hidden_states * 2, activation = "relu")
        self.GraphSage2  = GraphSageConv(hidden_states * 4, activation = "relu")

        self.Pool1   = GlobalMaxPool()
        self.Pool2   = GlobalAvgPool()
        self.Pool3   = GlobalSumPool()

        self.decode = [Dense(size * hidden_states) for size in [6, 6, 3]]
        self.drop_layers  = [Dropout(dropout) for i in range(len(self.decode))]
        self.norm_layers  = [BatchNormalization() for i in range(len(self.decode))]

        self.angles     = [Dense(hidden_states) for i in range(2)]
        self.angles_out = Dense(n_out)

        if n_kappa > 0:
          self.sigs      = [Dense(hidden_states) for i in range(2)]
          self.sigs_out  = Dense(n_kappa)

        if n_corr > 0:

          self.corr      = [Dense(hidden_states) for i in range(2)]
          self.corr_out  =  Dense(n_corr)



    def call(self, inputs, training = False):
        x, a, i = inputs
        x       = self.normalize(x)
        a, e    = self.generate_edge_features(x, a)
        e       = self.batch_edge(e)
        x = self.MP([x, a, e])
        x = self.GraphSage1([x, a])
        x = self.GraphSage2([x, a])

        x1 = self.Pool1([x, i])
        x2 = self.Pool2([x, i])
        x3 = self.Pool3([x, i])
        x = tf.concat([x1, x2, x3], axis = 1)

        for decode_layer, norm_layer, drop_layer in zip(self.decode, self.norm_layers, self.drop_layers):
          x = drop_layer(x, training = training)
          x = activation(decode_layer(x))
          x = norm_layer(x, training = training)

        x_angles = self.angles[0](x)
        x_angles = self.angles[1](x_angles)
        x_angles = self.angles_out(x_angles)

        if self.cossin:
          x_angles = tf.math.tanh(x_angles)


        if self.n_sigs > 0:
          x_sigs  = self.sigs[0](x)
          x_sigs  = self.sigs[1](x_sigs)
          x_sigs  = tf.abs(self.sigs_out(x_sigs)) + eps

        if self.n_corr > 0:
          x_corr = self.corr[0](x)
          x_corr = self.corr[1](x_corr)
          x_corr = tf.math.tanh(self.corr_out(x_corr))

          return tf.concat([x_angles, x_sigs, x_corr], axis = 1)
        
        if self.n_sigs > 0:
          return tf.concat([x_angles, x_sigs], axis = 1)
        else:
          return x_angles


        
    def normalize(self, input):
      input -= self.norm_trans
      input /= self.norm_scale
      return input

    def generate_edge_features(self, x, a):
      send    = a.indices[:, 0]
      receive = a.indices[:, 1]
      
      if self.forward == True:
        forwards  = tf.gather(x[:, 3], send) <= tf.gather(x[:, 3], receive)

        send    = tf.cast(send[forwards], tf.int64)
        receive = tf.cast(receive[forwards], tf.int64)

        a       = SparseTensor(indices = tf.stack([send, receive], axis = 1), values = tf.ones(tf.shape(send), dtype = tf.float32), dense_shape = tf.cast(tf.shape(a), tf.int64))

      diff_x  = tf.subtract(tf.gather(x, receive), tf.gather(x, send))

      dists   = tf.sqrt(
        tf.reduce_sum(
          tf.square(
            diff_x[:, :3]
          ), axis = 1
        ))

      vects = tf.math.divide_no_nan(diff_x[:, :3], tf.expand_dims(dists, axis = -1))

      e = tf.concat([diff_x[:, 3:], tf.expand_dims(dists, -1), vects], axis = 1)

      return a, e




class GraphSage_network(Model):
    def __init__(self, n_out = 3, n_kappa = 1, n_corr = 0, n_in = 5, hidden_states = 64, forward = False, dropout = 0, gamma = 0, **kwargs):
        super().__init__()
        self.forward = forward
        self.norm_trans   = normalize6['translate'][:n_in]
        self.norm_scale   = normalize6['scale'][:n_in]
        self.gamma        = gamma
        self.n_corr       = n_corr
        self.n_kappa      = n_kappa

        self.batch_edge  = BatchNormalization()

        self.MP          = MP(hidden_states, hidden_states, dropout = dropout)

        self.GraphSage1  = GraphSageConv(hidden_states * 2, activation = "relu")
        self.GraphSage2  = GraphSageConv(hidden_states * 4, activation = "relu")

        self.Pool1   = GlobalMaxPool()
        self.Pool2   = GlobalAvgPool()
        self.Pool3   = GlobalSumPool()

        self.decode = [Dense(size * hidden_states) for size in [6, 6, 3]]
        self.drop_layers  = [Dropout(dropout) for i in range(len(self.decode))]
        self.norm_layers  = [BatchNormalization() for i in range(len(self.decode))]

        self.units     = [Dense(hidden_states) for i in range(2)]
        self.units_out = Dense(n_out)

        if self.n_kappa > 0 :
          self.sigs      = [Dense(hidden_states) for i in range(2)]

          self.sigs_out  = Dense(n_kappa)

        if n_corr > 0:

          self.corr      = [Dense(hidden_states) for i in range(2)]
          self.corr_out  =  Dense(n_corr)



    def call(self, inputs, training = False):
        x, a, i = inputs
        x       = self.normalize(x)
        a, e    = self.generate_edge_features(x, a)
        e       = self.batch_edge(e)
        x = self.MP([x, a, e])
        x = self.GraphSage1([x, a])
        x = self.GraphSage2([x, a])

        x1 = self.Pool1([x, i])
        x2 = self.Pool2([x, i])
        x3 = self.Pool3([x, i])
        x = tf.concat([x1, x2, x3], axis = 1)

        for decode_layer, norm_layer, drop_layer in zip(self.decode, self.norm_layers, self.drop_layers):
          x = drop_layer(x, training = training)
          x = activation(decode_layer(x))
          x = norm_layer(x, training = training)

        x_units = self.units[0](x)
        x_units = self.units[1](x_units)
        x_units = self.units_out(x_units)

        x_norm   = tf.math.reduce_euclidean_norm(x_units, axis = 1)

        self.add_loss(tf.reduce_mean(self.gamma * abs(x_norm - 1)))

        x_units = tf.math.divide_no_nan(x_units, tf.expand_dims(x_norm, axis = -1))
        if self.n_kappa > 0:
          x_sigs  = self.sigs[0](x)
          x_sigs  = self.sigs[1](x_sigs)
          x_sigs  = tf.abs(self.sigs_out(x_sigs)) + eps

        if self.n_corr > 0:
          x_corr = self.corr[0](x)
          x_corr = self.corr[1](x_corr)
          x_corr = tf.math.tanh(self.corr_out(x_corr))

          return tf.concat([x_units, x_sigs, x_corr], axis = 1)

        if self.n_kappa > 0:
          return tf.concat([x_units, x_sigs], axis = 1)
        else:
          return x_units

    def normalize(self, input):
      input -= self.norm_trans
      input /= self.norm_scale
      return input

    def generate_edge_features(self, x, a):
      send    = a.indices[:, 0]
      receive = a.indices[:, 1]
      
      if self.forward == True:
        forwards  = tf.gather(x[:, 3], send) <= tf.gather(x[:, 3], receive)

        send    = tf.cast(send[forwards], tf.int64)
        receive = tf.cast(receive[forwards], tf.int64)

        a       = SparseTensor(indices = tf.stack([send, receive], axis = 1), values = tf.ones(tf.shape(send), dtype = tf.float32), dense_shape = tf.cast(tf.shape(a), tf.int64))

      diff_x  = tf.subtract(tf.gather(x, receive), tf.gather(x, send))

      dists   = tf.sqrt(
        tf.reduce_sum(
          tf.square(
            diff_x[:, :3]
          ), axis = 1
        ))

      vects = tf.math.divide_no_nan(diff_x[:, :3], tf.expand_dims(dists, axis = -1))

      e = tf.concat([diff_x[:, 3:], tf.expand_dims(dists, -1), vects], axis = 1)

      return a, e


class MP(MessagePassing):

    def __init__(self, n_out, hidden_states, dropout = 0):
        super().__init__()
        self.n_out = n_out
        self.hidden_states = hidden_states
        self.message_mlp = MLP(hidden_states * 2, hidden = hidden_states * 4, layers = 2, dropout = dropout)
        self.update_mlp  = MLP(hidden_states * 1, hidden = hidden_states * 2, layers = 2, dropout = dropout)

    def propagate(self, x, a, e=None, training = False, **kwargs):
        self.n_nodes = tf.shape(x)[0]
        self.index_i = a.indices[:, 1]
        self.index_j = a.indices[:, 0]

        # Message
        # print(x, a, e)
        # msg_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs)
        messages = self.message(x, a, e, training = training)

        # Aggregate
        # agg_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)
        embeddings = self.aggregate(messages, training = training)

        # Update
        # upd_kwargs = self.get_kwargs(x, a, e, self.upd_signature, kwargs)
        output = self.update(embeddings, training = training)

        return output

    def message(self, x, a, e, training = False):
        # print([self.get_i(x), self.get_j(x), e])
        out = tf.concat([self.get_i(x), self.get_j(x), e], axis = 1)
        out = self.message_mlp(out, training = training)
        return out
    
    def update(self, embeddings, training = False):
        out = self.update_mlp(embeddings, training = training)
        return out

class MLP(Model):
    def __init__(self, output, hidden=256, layers=2, batch_norm=True,
                 dropout=0.0, activation='relu', final_activation=None):
        super().__init__()
        self.batch_norm = batch_norm
        self.dropout_rate = dropout

        self.mlp = Sequential()
        for i in range(layers):
            # Linear
            self.mlp.add(Dense(hidden if i < layers - 1 else output, activation = activation))
            if dropout > 0:
                self.mlp.add(Dropout(dropout))


    def call(self, inputs, training = False):
        return self.mlp(inputs, training = training)

        
        