import numpy as np

from keras.layers import Layer
from keras.callbacks import Callback

import keras.backend as K

class ConstSqLayer(Layer):
  def __init__(self, **kwargs):
    super(ConstSqLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    self.sig = self.add_weight(name='sigma', shape=(1,), initializer=Constant(value=1), trainable=True)
    super(ConstSqLayer, self).build(input_shape)

  def call(self, x):
    const = K.ones_like(x)
    sig_arr = self.sig*const
    return K.square(sig_arr)

  def compute_output_shape(self, input_shape):
    return input_shape
    
class SimpleCallback(Callback):
  def __init__(self):
    pass
  
  def on_epoch_end(self, epoch, logs={}):
    print('\rEpoch: {}'.format(epoch), end='')
    
class ProportionCallback(Callback):
  """
  Assumes output is in the form of a categorical distribution over n_classes.
  """
  def __init__(self, valid_x, n_classes):
    self.valid_x = valid_x
    self.n_classes = n_classes
    self.proportions = [[] for k in range(n_classes)]

  def on_train_begin(self, logs={}):
    out = self.model.predict(self.valid_x)
    for k in range(self.n_classes):
      self.proportions[k].append(out[:,k].mean())

  def on_epoch_end(self, epoch, logs={}):
    print('\rEpoch: {}'.format(epoch), end='')
    out = self.model.predict(self.valid_x)
    for k in range(self.n_classes):
      self.proportions[k].append(out[:,k].mean())
    
def noise_loss_factory(sig2, original_loss):
  def loss(y_true, y_pred):
    return original_loss(y_true, y_pred)/sig2 + K.log(sig2)
  return loss
  
class ClusteringEntropyCallback(Callback):
  """
  For a multi-task network where each task outputs a categorical distribution with
    2, 3, 4, ..., n_clusters categories. Records the entropy of the mean over a 
    validation set after each epoch for each task.
  """
  def __init__(self, valid_x, n_clusters):
    self.valid_x = valid_x
    self.n_clusters = n_clusters
    self.clustering_entropy = [[] for k in range(n_clusters-1)]

  def get_cluster_entropies(self):
    out = self.model.predict(self.valid_x)

    clustering_entropies = []
    for k in range(self.n_clusters-1):
      p = out[k].mean(axis=0)
      ent = (-p*np.log(p)).sum()
      clustering_entropies.append(np.log(k+2)-ent)

    return clustering_entropies
    
  def on_train_begin(self, logs={}):
    for k, ce in enumerate(self.get_cluster_entropies()):
      self.clustering_entropy[k].append(ce)

  def on_epoch_end(self, epoch, logs={}):
    print('\rEpoch: {}'.format(epoch), end='')
    for k, ce in enumerate(self.get_cluster_entropies()):
      self.clustering_entropy[k].append(ce)
  
def get_clusters(n, clusters=list('abc'), dist=5, cluster_weight=None):
  """
  Generates n samples from equidistant clusters in a 2d plane.
  """
  cluster_shift_dict = {c:np.array([np.cos(2*k*np.pi/len(clusters)), np.sin(2*k*np.pi/len(clusters))]) for k, c in enumerate(clusters)}
  if cluster_weight is None:
    cluster_id = np.random.choice(clusters, n)
  else:
    cluster_id = np.random.choice(clusters, n, p=np.array(cluster_weight)/sum(cluster_weight))
  cluster_shift = np.array([cluster_shift_dict[c] for c in cluster_id])
  
  x = np.random.randn(n,2) + dist*cluster_shift

  return x

def mixed_entropy_loss(y_true, y_pred):
  """
  Keras loss, equal to the mean entropy of a sample plus 
    KL-Divergence of the mean of the batch viewed as a categorical distribution
    from the uniform distribution.
  """
  eps = 1e-10

  mean = K.mean(y_pred, axis=0)
  c = K.ones_like(mean)
  c = c/K.sum(c)
  max_entropy = -K.sum(c*K.log(c+eps))

  nentropy_mean = K.mean(K.sum(y_pred*K.log(y_pred+eps), axis=1))
  mean_nentropy = K.sum(mean*K.log(mean+eps))

  return mean_nentropy - nentropy_mean + max_entropy
