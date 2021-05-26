import os
from spektral.layers.convolutional.gcn_conv import GCNConv

import tensorflow as tf

from spektral.layers import CrystalConv, GraphSageConv, MessagePassing, GATConv, GatedGraphConv
from spektral.layers.pooling.global_pool import GlobalMaxPool, GlobalAvgPool, GlobalSumPool

from tensorflow.keras import Model, Input, Sequential 
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.activations import tanh
from tensorflow.sparse import SparseTensor
from tensorflow.keras.backend import epsilon

import numpy as np

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

normalize6 ={"translate": tf.constant([0, 0, -200, 10000, 0, 1.], dtype = tf.float32),
             "scale":     tf.constant([100, 100, 100, 2500, 0.25, 7.], dtype = tf.float32),
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
          x = activation(decode_layer(x))
          x = drop_layer(x, training = training)
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


class GraphSage_network2(Model):
    def __init__(self, n_out = 3, n_kappa = 1, n_corr = 0, n_in = 5, hidden_states = 64, \
                decode_layers = [6, 6, 3], mp_layers = 1,\
                forward = False, dropout = 0, gamma = 0, sigmoid = False, scale = True, **kwargs):
        super().__init__()
        self.forward = forward
        self.norm_trans   = normalize6['translate'][:n_in]
        self.norm_scale   = normalize6['scale'][:n_in]
        self.gamma        = gamma
        self.n_out        = n_out
        self.n_corr       = n_corr
        self.n_kappa      = n_kappa
        self.sigmoid      = sigmoid
        self.scale        = scale

        if self.scale:
          self.batch_edge  = BatchNormalization()

        self.MP          = [MP2(hidden_states, dropout = dropout) for i in range(mp_layers)]

        self.GraphSage1  = GraphSageConv(hidden_states * 2, activation = "relu")
        self.GraphSage2  = GraphSageConv(hidden_states * 4, activation = "relu")

        self.Pool1   = GlobalMaxPool()
        self.Pool2   = GlobalAvgPool()
        self.Pool3   = GlobalSumPool()
        self.Pool4   = GlobalMaxPool()

        self.decode = [Dense(size * hidden_states) for size in decode_layers]
        self.drop_layers  = [Dropout(dropout) for i in range(len(self.decode))]
        self.norm_layers  = [BatchNormalization() for i in range(len(self.decode))]

        self.outs    = []
        for i in range(self.n_out):
          mlp_out = [Dense(hidden_states) for i in range(2)]
          mlp_out.append(Dense(1))
          self.outs.append(mlp_out)

        if self.n_kappa > 0 :
          self.sigs      = [Dense(hidden_states) for i in range(2)]

          self.sigs_out  = Dense(n_kappa)

        if n_corr > 0:

          self.corr      = [Dense(hidden_states) for i in range(2)]
          self.corr_out  =  Dense(n_corr)



    def call(self, inputs, training = False):
        x, a, i = inputs
        if self.scale:
          x       = self.normalize(x)

        a, e    = self.generate_edge_features(x, a)
        if self.scale:
          e       = self.batch_edge(e)
        
        for MP_layer in self.MP:
          x = MP_layer([x, a, e])
        x = self.GraphSage1([x, a])
        x = self.GraphSage2([x, a])

        x1 = self.Pool1([x, i])
        x2 = self.Pool2([x, i])
        x3 = self.Pool3([x, i])
        x4 = -1 * self.Pool4([-1 * x, i])
        x = tf.concat([x1, x2, x3, x4], axis = 1)

        for decode_layer, norm_layer, drop_layer in zip(self.decode, self.norm_layers, self.drop_layers):
          x = activation(decode_layer(x))
          x = drop_layer(x, training = training)
          x = norm_layer(x, training = training)

        out = []
        for layer_sets in self.outs:
          o = x
          for layer in layer_sets:
            o  = layer(o)
            out.append(o)
        
        x_units = tf.concat(out, axis = 1)

        if self.n_kappa > 0:
          x_sigs  = self.sigs[0](x)
          x_sigs  = self.sigs[1](x_sigs)
          x_sigs  = tf.abs(self.sigs_out(x_sigs)) + eps


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


class GraphSage_network(Model):
    def __init__(self, n_out = 3, n_kappa = 1, n_corr = 0, n_in = 5, hidden_states = 64, \
                decode_layers = [6, 6, 3], mp_layers = 1, normalize_output = True,\
                forward = False, dropout = 0, gamma = 0, sigmoid = False, scale = True, **kwargs):
        super().__init__()
        self.forward = forward
        self.norm_trans   = normalize6['translate'][:n_in]
        self.norm_scale   = normalize6['scale'][:n_in]
        self.gamma        = gamma
        self.n_out        = n_out
        self.n_corr       = n_corr
        self.n_kappa      = n_kappa
        self.sigmoid      = sigmoid
        self.scale        = scale
        self.normalize_output = normalize_output

        if self.scale:
          self.batch_edge  = BatchNormalization()

        self.MP          = [MP2(hidden_states, dropout = dropout) for i in range(mp_layers)]

        self.GraphSage1  = GraphSageConv(hidden_states * 2, activation = "relu")
        self.GraphSage2  = GraphSageConv(hidden_states * 4, activation = "relu")

        self.Pool1   = GlobalMaxPool()
        self.Pool2   = GlobalAvgPool()
        self.Pool3   = GlobalSumPool()
        self.Pool4   = GlobalMaxPool()

        self.decode = [Dense(size * hidden_states) for size in decode_layers]
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
        if self.scale:
          x       = self.normalize(x)

        a, e    = self.generate_edge_features(x, a)
        if self.scale:
          e       = self.batch_edge(e, training = training)
        
        for MP_layer in self.MP:
          x = MP_layer([x, a, e], training = training)
        x = self.GraphSage1([x, a])
        x = self.GraphSage2([x, a])

        x1 = self.Pool1([x, i])
        x2 = self.Pool2([x, i])
        x3 = self.Pool3([x, i])
        x4 = -1 * self.Pool4([-1 * x, i])
        x = tf.concat([x1, x2, x3, x4], axis = 1)

        for decode_layer, norm_layer, drop_layer in zip(self.decode, self.norm_layers, self.drop_layers):
          x = activation(decode_layer(x))
          x = drop_layer(x, training = training)
          x = norm_layer(x, training = training)

        x_units = self.units[0](x)
        x_units = self.units[1](x_units)
        x_units = self.units_out(x_units)

        if self.normalize_output:
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



class MessagePassModel(Model):

    def __init__(self, n_out = 3, n_kappa = 1, n_in = 5, hidden_states = 64, dropout = 0, batch_norm = True, mp_layers = 3,\
                 message_size = 2, message_layers = 2, update_size = 4, update_layers = 2, decode_layers = [6, 6, 3], \
                    split_structure = [1, 1], sigmoid = False, scale = True, convs = False, units_out = True, **kwargs):
        
        # Setups 
        super().__init__()
        self.norm_trans   = normalize6['translate'][:n_in]
        self.norm_scale   = normalize6['scale'][:n_in]
        self.n_out        = n_out
        self.n_kappa      = n_kappa
        self.sigmoid      = sigmoid
        self.scale        = scale 
        self.units_out    = units_out

        self.batch_edge  = BatchNormalization()
        
        self.apply_layers = []
        self.bn_layers    = []
        for i in range(mp_layers):
          self.apply_layers.append(MP2(hidden_states = hidden_states,  message_size = message_size,   message_layers = message_layers, \
                                        update_size   = update_size,    update_layers = update_layers, dropout = dropout))
          if batch_norm:
            self.bn_layers.append(BatchNormalization())
        
        if convs:
          self.convs = []
          if convs == "attention":
            self.convs.append(GATConv(hidden_states * 2, attn_heads = 4, dropout_rate = dropout))
            self.convs.append(Dropout(dropout))
            self.convs.append(BatchNormalization())
            self.convs.append(GATConv(hidden_states * 1, attn_heads = 2, dropout_rate = dropout))
            self.convs.append(Dropout(dropout))
            self.convs.append(BatchNormalization())
          elif convs == "GRU":
            self.convs.append(GatedGraphConv(hidden_states, n_layers = 5))
            self.convs.append(Dropout(dropout))
            self.convs.append(BatchNormalization())
            self.convs.append(GatedGraphConv(hidden_states, n_layers = 3))
            self.convs.append(Dropout(dropout))
            self.convs.append(BatchNormalization())
          elif convs == "GraphSage":
            self.convs.append(GraphSageConv(hidden_states * 4, activation = "gelu"))
            self.convs.append(Dropout(dropout))
            self.convs.append(BatchNormalization())
            self.convs.append(GraphSageConv(hidden_states * 2, activation = "gelu"))
            self.convs.append(Dropout(dropout))
            self.convs.append(BatchNormalization())
        else:
          print("No convulutions defined.")
          self.convs = None

        self.Pool1   = GlobalMaxPool()
        self.Pool2   = GlobalAvgPool()
        self.Pool3   = GlobalSumPool()
        self.Pool4   = GlobalMaxPool()

        self.decode_layers = []
        for i in decode_layers:
          self.decode_layers.append(Dense(i * hidden_states))
          self.decode_layers.append(Dropout(dropout))
          if batch_norm:
            self.decode_layers.append(BatchNormalization())
        
        self.split_layers = []
        for i in range(n_out + n_kappa):
          split = []
          for j in split_structure:
            split.append(Dense(hidden_states * j))
            split.append(Dropout(dropout))
            split.append(BatchNormalization())
          split.append(Dense(1))
          self.split_layers.append(split)


    def call(self, inputs, training = False):
        x, a, i = inputs
        x       = self.normalize(x)
        a, e    = self.generate_edge_features(x, a)
        e       = self.batch_edge(e, training = training)

        for MessagePassLayer, BatchNormLayer in zip(self.apply_layers, self.bn_layers):
          x = MessagePassLayer([x, a, e], training = training)
          x = BatchNormLayer(x, training = training)
        
        if self.convs:
          for layer in self.convs:
              if isinstance(layer, Dropout) or isinstance(layer, BatchNormalization):
                x_out = layer(x, training = training)
              else:
                x = layer([x, a], training = training)

        
        x1 = self.Pool1([x, i])
        x2 = self.Pool2([x, i])
        x3 = self.Pool3([x, i])
        x4 = -1 * self.Pool4([-1 * x, i])
        x = tf.concat([x1, x2, x3, x4], axis = 1)
        
        for DecodeLayer in self.decode_layers:
          if isinstance(DecodeLayer, Dense):
            x = DecodeLayer(x)
          else:
            x = DecodeLayer(x, training = training)

        output = []
        for i, split in enumerate(self.split_layers):
          x_out = split[0](x)

          if len(split) > 1:
            for layer in split[1:]:
              if isinstance(layer, Dropout) or isinstance(layer, BatchNormalization):
                x_out = layer(x_out, training = training)
              else:
                x_out = layer(x_out)
          
          output.append(x_out)
        
        x = tf.concat(output, axis = 1)

        if self.n_out == 4 and self.units_out:
          x_norm   = tf.math.reduce_euclidean_norm(x[:, :3], axis = 1)
          x_out = tf.math.divide_no_nan(x[:, :3], tf.expand_dims(x_norm, axis = -1))
          x_out = tf.concat([x_out, tf.expand_dims(x[:, 3], axis = 1)], axis = 1)
        elif self.n_out == 3 and self.units_out:
          x_norm   = tf.math.reduce_euclidean_norm(x[:, :3], axis = 1)
          x_out = tf.math.divide_no_nan(x[:, :3], tf.expand_dims(x_norm, axis = -1))
        else:
          x_out = x[:, :self.n_out]
        
        if self.n_out == 1 and self.sigmoid:
          x_out = tf.math.sigmoid(x)

        if self.n_kappa == 0:
          return x_out
        else:
          x_kappa = tf.abs(x[:, - self.n_kappa:]) + eps 
          return tf.concat([x_out, x_kappa], axis = 1)




    def normalize(self, input):
      input -= self.norm_trans
      input /= self.norm_scale
      return input

    def generate_edge_features(self, x, a):
      send    = a.indices[:, 0]
      receive = a.indices[:, 1]
      
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
        self.index_i = a.indices[:, 0]
        self.index_j = a.indices[:, 1]

        # Message
        # print(x, a, e)
        # msg_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs)
        messages = self.message(x, a, e, training = training)

        # Aggregate
        # agg_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)
        embeddings = self.describe(messages, self.index_i)

        # Update
        # upd_kwargs = self.get_kwargs(x, a, e, self.upd_signature, kwargs)
        output = self.update(embeddings, training = training)

        return output
    

    def describe(self, input, index):
        minimum = tf.math.segment_min(input, index)
        maximum = tf.math.segment_max(input, index)
        mean    = tf.math.segment_mean(input, index)
        var     = tf.math.segment_mean(input ** 2, index) - tf.math.segment_mean(input, index) ** 2

        return tf.concat([minimum, maximum, mean, var], axis = 1)


    def message(self, x, a, e, training = False):
        # print([self.get_i(x), self.get_j(x), e])
        out = tf.concat([self.get_i(x), self.get_j(x), e], axis = 1)
        out = self.message_mlp(out, training = training)
        return out
    
    def update(self, embeddings, training = False):
        out = self.update_mlp(embeddings, training = training)
        return out


class MP2(MessagePassing):

    def __init__(self,            hidden_states,     message_size = 2,   message_layers = 2, \
                                  update_size  = 4,  update_layers = 2,  dropout        = 0):
        super().__init__()
        self.hidden_states = hidden_states
        self.message_mlp = MLP(message_size, message_size, layers = message_layers, dropout = dropout)
        self.update_mlp  = MLP(update_size,  update_size , layers = update_layers,  dropout = dropout)


    def propagate(self, x, a, e=None, training = False, **kwargs):
        self.n_nodes = tf.shape(x)[0]
        self.index_i = a.indices[:, 0]
        self.index_j = a.indices[:, 1]

        # Message
        messages = self.message(x, a, e, training = training)

        # Aggregate
        embeddings = self.describe(messages, self.index_i)

        # Update
        output = self.update(embeddings, training = training)

        return output
    

    def describe(self, input, index):
        minimum = tf.math.segment_min(input, index)
        maximum = tf.math.segment_max(input, index)
        mean    = tf.math.segment_mean(input, index)
        var     = tf.math.segment_mean(input ** 2, index) - tf.math.segment_mean(input, index) ** 2

        return tf.concat([minimum, maximum, mean, var], axis = 1)


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

        
        



# from tensorflow.math import multiply
from tensorflow.keras.activations import sigmoid



def no_norm(x, training):
  return x

class KHop(Model):
    def __init__(self, n_out = 2, n_sigs=2, K=[1,2], agg_method='sum', hidden_states=32, glob=True, conv_layers=1, conv_activation='relu', decode_layers=2, decode_activation=1, regularization=None, dropout=0.2, batch_norm=True, forward=True):
        super().__init__()
        self.n_out=n_out
        self.n_sigs=n_sigs
        self.hidden_states=hidden_states
        self.conv_activation=conv_activation
        self.forward=forward
        self.dropout=dropout
        self.glob=glob
        self.Ks=K
        self.agg_method=agg_method
        self.conv_layers=conv_layers
        self.regularize=regularization
        if type(decode_activation)==str:
          self.decode_activation=tf.keras.activations.get(decode_activation)
        else:
          self.decode_activation=activation
        self.batch_norm=batch_norm
        # Define layers of the model

        self.MPs      = [SGConv(hidden_states, hidden_states, K=K, agg_method=self.agg_method, dropout = dropout) for K in self.Ks]

        self.GCNs    = [GraphSageConv(hidden_states*int(i), activation=self.conv_activation, kernel_regularizer=self.regularize) for i in 2*2**np.arange(self.conv_layers)]

        self.Pool1   = GlobalMaxPool()
        self.Pool2   = GlobalAvgPool()
        self.Pool3   = GlobalSumPool()

        self.decode  = [Dense(i * hidden_states) for i in  2*2**np.arange(decode_layers+1,1,-1)]
        self.dropout_layers  = [Dropout(dropout) for i in range(len(self.decode))]
        if self.batch_norm:
          self.norm_layers  = [BatchNormalization() for i in range(len(self.decode))]
        else:
          self.norm_layers =  [no_norm for i in range(len(self.decode))]
        
        self.loge     = [Dense(hidden_states) for _ in range(2)]
        self.loge_out = Dense(1)
        self.angles     = [Dense(hidden_states) for _ in range(2)]
        self.angles_out = Dense(1)
        self.angle_scale= Dense(1)
        if n_sigs > 0:
          self.sigs      = [Dense(hidden_states) for i in range(2)]
          self.sigs_out  = Dense(n_sigs)

    def call(self, inputs, training = False):
        x, a, i = inputs
        glob_avg=tf.math.segment_mean(x,i)
        glob_var=abs(tf.math.subtract(tf.math.segment_mean(x * x, i), glob_avg * glob_avg))
        glob_max=tf.math.segment_max(x,i)
        glob_min=tf.math.segment_min(x,i)
        xglob=tf.concat([glob_avg, glob_var, glob_max, glob_min], axis=1)
        a, e    = self.generate_edge_features(x, a)
        for MP in self.MPs:
          x = MP([x, a, e])
        for conv in self.GCNs:
          x=conv([x,a])
        x1 = self.Pool1([x, i])
        x2 = self.Pool2([x, i])
        x3 = self.Pool3([x, i])
        x = tf.concat([x1, x2, x3], axis = 1)
        x=tf.concat([x, xglob], axis=1)
        for decode_layer, dropout_layer, norm_layer in zip(self.decode, self.dropout_layers, self.norm_layers):
          x = dropout_layer(x, training = training)
          x = self.decode_activation(decode_layer(x))
          x = norm_layer(x, training = training)
                
        x_loge = self.loge[0](x)
        x_loge = self.loge[1](x_loge)
        x_loge = self.loge_out(x_loge)

        x_angles = self.angles[0](x)
        x_angles = self.angles[1](x_angles)
        x_angles = self.angles_out(x_angles)
        zeniazi=sigmoid(self.angle_scale(x_angles))

        if self.n_sigs > 0:
          x_sigs  = self.sigs[0](x)
          x_sigs  = self.sigs[1](x_sigs)
          x_sigs  = tf.abs(self.sigs_out(x_sigs)) + eps
        #could add correlation here 
        xs=tf.concat([x_loge, zeniazi*np.pi], axis = 1)
        if self.n_sigs > 0:
          return tf.concat([xs, x_sigs], axis=1)
        else:
          return xs


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


class SGConv(MessagePassing):
    # note that the D^-1/2 norm is not implemented since it is irrelevant for us 
    def __init__(self, n_out, hidden_states, K=2, agg_method='sum', dropout = 0):

        """Agg_method supports "sum": scatter_sum,
          "mean": scatter_mean,
          "max": scatter_max,
          "min": scatter_min,
          "prod": scatter_prod"""
        super().__init__()
        self.n_out = n_out
        self.agg_method=agg_method
        self.K=K
        self.hidden_states = hidden_states
        self.message_mlps = [MLP(hidden_states, hidden = hidden_states * 2, layers = 2, dropout = dropout) for _ in range(self.K)]
        self.update_mlp  = MLP(hidden_states, hidden = hidden_states, layers = 2, dropout = dropout)


    ##inverted structure since tf requires output func to be propagate
    def prop_khop(self, x, a, k, e=None, training = False, **kwargs):
        self.n_nodes = tf.shape(x)[0]
        self.index_i = a.indices[:, 1]
        self.index_j = a.indices[:, 0]

        # Message
        # print(x, a, e)
        # msg_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs)
        messages = self.message(x, a, k, e, training = training)

        # Aggregate
        # agg_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)

        ##  make own aggregate
        embeddings = self.aggregate(messages, training = training)

        return embeddings

    def propagate(self, x, a, e, training=False):
        for hop in range(self.K):
          x=self.prop_khop(x,a, hop, e, training = training)
        return self.update(x, training = training)

    def message(self, x, a, k, e, training = False):
        # print([self.get_i(x), self.get_j(x), e])
        out = tf.concat([self.get_i(x), self.get_j(x), e], axis = 1)
        out = self.message_mlps[k](out, training = training)
        return out
    
    def update(self, embeddings, training = False):
        out = self.update_mlp(embeddings, training = training)
        return out 