import oneflow as flow
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import math

class BertBackbone(object):

  def __init__(self,
               input_ids_blob,
               input_mask_blob,
               token_type_ids_blob,
               vocab_size,
               seq_length=512,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):

    dl_net = input_ids_blob.dl_net()
    assert dl_net == input_mask_blob.dl_net()
    assert dl_net == token_type_ids_blob.dl_net()
    with dl_net.VariableScope("bert"):
      with dl_net.VariableScope("embeddings"):
        (self.embedding_output_, self.embedding_table_) = _EmbeddingLookup(
            input_ids_blob=input_ids_blob,
            vocab_size=vocab_size,
            embedding_size=hidden_size,
            initializer_range=initializer_range,
            word_embedding_name="word_embeddings")
        self.embedding_output_ = _EmbeddingPostprocessor(
            input_blob=self.embedding_output_,
            seq_length=seq_length,
            embedding_size=hidden_size,
            use_token_type=True,
            token_type_ids_blob=token_type_ids_blob,
            token_type_vocab_size=type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=initializer_range,
            max_position_embeddings=max_position_embeddings,
            dropout_prob=hidden_dropout_prob)
      with dl_net.VariableScope("encoder"):
        attention_mask_blob = _CreateAttentionMaskFromInputMask(
          input_mask_blob, from_seq_length=seq_length, to_seq_length=seq_length)
        self.all_encoder_layers_ = _TransformerModel(
            input_blob=self.embedding_output_,
            attention_mask_blob=attention_mask_blob,
            seq_length=seq_length,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            intermediate_act_fn=GetActivation(hidden_act),
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_all_layers=False)
      self.sequence_output_ = self.all_encoder_layers_[-1]

  def embedding_output(self): return self.embedding_output_
  def all_encoder_layers(self): return self.all_encoder_layers_
  def sequence_output(self): return self.sequence_output_
  def embedding_table(self): return self.embedding_table_

def CreateInitializer(std):
  return {'truncated_normal_conf': {'std':std}}

def Reshape(in_blob, shape):
  dl_net = in_blob.dl_net()
  return dl_net.Reshape(
      in_blob,
      shape={'dim': shape},
      has_dim0_in_shape=True)

def _Gelu(in_blob):
  dl_net = in_blob.dl_net()
  return dl_net.Gelu(in_blob)

def _TransformerModel(input_blob,
                      attention_mask_blob,
                      seq_length,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=_Gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):

  assert hidden_size % num_attention_heads == 0
  dl_net = input_blob.dl_net()
  attention_head_size = int(hidden_size / num_attention_heads)
  input_width = hidden_size
  prev_output_blob = Reshape(input_blob, (-1, input_width))
  all_layer_output_blobs = []
  for layer_idx in range(num_hidden_layers):
    with dl_net.VariableScope("layer_%d"%layer_idx):
      layer_input_blob = prev_output_blob
      with dl_net.VariableScope("attention"):
        with dl_net.VariableScope("self"):
          attention_output_blob = _AttentionLayer(
              from_blob=layer_input_blob,
              to_blob=layer_input_blob,
              attention_mask_blob=attention_mask_blob,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
        with dl_net.VariableScope("output"):
          attention_output_blob = _FullyConnected(
              attention_output_blob,
              input_size=num_attention_heads * attention_head_size,
              units=hidden_size,
              weight_initializer=CreateInitializer(initializer_range),
              name='dense')
          attention_output_blob = _Dropout(attention_output_blob, hidden_dropout_prob)
          attention_output_blob = dl_net.Add([attention_output_blob, layer_input_blob])
          attention_output_blob = _LayerNorm(attention_output_blob, hidden_size)
      with dl_net.VariableScope("intermediate"):
        if callable(intermediate_act_fn):
          act_fn = op_conf_util.kNone
        else:
          act_fn = intermediate_act_fn
        intermediate_output_blob = _FullyConnected(
            attention_output_blob,
            input_size=num_attention_heads * attention_head_size,
            units=intermediate_size,
            activation=act_fn,
            weight_initializer=CreateInitializer(initializer_range),
            name='dense')
        if callable(intermediate_act_fn):
          intermediate_output_blob = intermediate_act_fn(intermediate_output_blob)
      with dl_net.VariableScope("output"):
        layer_output_blob = _FullyConnected(
            intermediate_output_blob,
            input_size=intermediate_size,
            units=hidden_size,
            weight_initializer=CreateInitializer(initializer_range),
            name='dense')
        layer_output_blob = _Dropout(layer_output_blob, hidden_dropout_prob)
        layer_output_blob = dl_net.Add([layer_output_blob, attention_output_blob])
        layer_output_blob = _LayerNorm(layer_output_blob, hidden_size)
        prev_output_blob = layer_output_blob
        all_layer_output_blobs.append(layer_output_blob)

  input_shape = (-1, seq_length, hidden_size)
  if do_return_all_layers:
    final_output_blobs = []
    for layer_output_blob in all_layer_output_blobs:
      final_output_blob = Reshape(layer_output_blob, input_shape)
      final_output_blobs.append(final_output_blob)
    return final_output_blobs
  else:
    final_output_blob = Reshape(prev_output_blob, input_shape)
    return [final_output_blob]

def _AttentionLayer(from_blob,
                    to_blob,
                    attention_mask_blob,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=op_conf_util.kNone,
                    key_act=op_conf_util.kNone,
                    value_act=op_conf_util.kNone,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):

  def TransposeForScores(input_blob, num_attention_heads, seq_length, width):
    dl_net = input_blob.dl_net()
    output_blob = Reshape(input_blob, [-1, seq_length, num_attention_heads, width])
    output_blob = dl_net.Transpose(output_blob, perm=[0, 2, 1, 3])
    return output_blob

  dl_net = from_blob.dl_net()
  from_blob_2d = Reshape(from_blob, [-1, num_attention_heads * size_per_head])
  to_blob_2d = Reshape(to_blob, [-1, num_attention_heads * size_per_head])

  query_blob = _FullyConnected(
      from_blob_2d,
      input_size=num_attention_heads * size_per_head,
      units=num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      weight_initializer=CreateInitializer(initializer_range))

  key_blob = _FullyConnected(
      to_blob_2d,
      input_size=num_attention_heads * size_per_head,
      units=num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      weight_initializer=CreateInitializer(initializer_range))

  value_blob = _FullyConnected(
      to_blob_2d,
      input_size=num_attention_heads * size_per_head,
      units=num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      weight_initializer=CreateInitializer(initializer_range))

  query_blob = TransposeForScores(query_blob, num_attention_heads, from_seq_length, size_per_head)
  key_blob = TransposeForScores(key_blob, num_attention_heads, to_seq_length, size_per_head)

  attention_scores_blob = dl_net.Matmul(query_blob, key_blob, transpose_b=True)
  attention_scores_blob = dl_net.ScalarMul(attention_scores_blob, 1.0 / math.sqrt(float(size_per_head)))

  attention_mask_blob = Reshape(attention_mask_blob, [-1, 1, from_seq_length, to_seq_length])
  attention_mask_blob = dl_net.Cast(attention_mask_blob, data_type=flow.float)
  addr_blob = dl_net.ScalarMul(dl_net.ScalarAdd(attention_mask_blob, -1.0), 10000.0)

  attention_scores_blob = dl_net.BroadcastAdd(attention_scores_blob, addr_blob)
  attention_probs_blob = dl_net.Softmax(attention_scores_blob)
  attention_probs_blob = _Dropout(attention_probs_blob, attention_probs_dropout_prob)

  value_blob = Reshape(value_blob, [-1, to_seq_length, num_attention_heads, size_per_head])
  value_blob = dl_net.Transpose(value_blob, perm=[0, 2, 1, 3])
  context_blob = dl_net.Matmul(attention_probs_blob, value_blob)
  context_blob = dl_net.Transpose(context_blob, perm=[0, 2, 1, 3])

  if do_return_2d_tensor:
    context_blob = Reshape(context_blob, [-1, num_attention_heads * size_per_head])
  else:
    context_blob = Reshape(context_blob, [-1, from_seq_length, num_attention_heads * size_per_head])
  return context_blob

def _FullyConnected(input_blob, input_size, units, activation=None, name=None,
                    weight_initializer=None):
  dl_net = input_blob.dl_net()
  weight_blob = dl_net.Variable(
    name=name + '-weight',
    shape={'dim': [input_size, units]},
    initializer=weight_initializer,
    model_name='weight')
  bias_blob = dl_net.Variable(
    name=name + '-bias',
    shape={'dim': [units]},
    initializer={'constant_conf': {'value':0.0}},
    model_name='bias')
  output_blob = dl_net.Matmul(input_blob, weight_blob)#, transpose_b=True)
  output_blob = dl_net.BiasAdd(output_blob, bias_blob)
  return output_blob
  #return dl_net.FullyConnected(input_blob, units=units, activation=activation, name=name,
  #    weight_initializer=weight_initializer)

def _Dropout(input_blob, dropout_prob):
  if dropout_prob == 0.0:
    return input_blob
  dl_net = input_blob.dl_net()
  return dl_net.Dropout(input_blob, rate=dropout_prob)


def _LayerNorm(input_blob, hidden_size, use_fused_layer_norm=True):
  dl_net = input_blob.dl_net()
  if use_fused_layer_norm:
    return dl_net.LayerNorm(input_blob, name="LayerNorm", begin_norm_axis=-1, begin_params_axis=-1)
  else:
    with dl_net.VariableScope('LayerNorm'):
      gamma_blob = dl_net.Variable(
        name='gamma',
        shape={'dim': [hidden_size]},
        initializer={'constant_conf': {'value': 1}},
        model_name='gamma')
      beta_blob = dl_net.Variable(
        name='beta',
        shape={'dim': [hidden_size]},
        initializer={'constant_conf': {'value': 0}},
        model_name='beta')
      shifted_input_blob = dl_net.BroadcastSub(input_blob, dl_net.Mean(input_blob))
      normalized_blob = dl_net.BroadcastDiv(
        shifted_input_blob,
        dl_net.Sqrt(dl_net.ScalarAdd(dl_net.Mean(dl_net.Square(shifted_input_blob)), 1e-12)))
      return dl_net.BroadcastAdd(dl_net.BroadcastMul(gamma_blob, normalized_blob), beta_blob)


def _CreateAttentionMaskFromInputMask(to_mask_blob, from_seq_length, to_seq_length):
  dl_net = to_mask_blob.dl_net()
  output = dl_net.Cast(to_mask_blob, data_type=flow.float)
  output = Reshape(output, [-1, 1, to_seq_length])
  zeros = dl_net.ConstantFill(0.0, shape=[from_seq_length, to_seq_length])
  output = dl_net.BroadcastAdd(zeros, output)
  return output


def _EmbeddingPostprocessor(input_blob,
                            seq_length,
                            embedding_size,
                            use_token_type=False,
                            token_type_ids_blob=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  dl_net = input_blob.dl_net()
  output = input_blob

  if use_token_type:
    assert token_type_ids_blob is not None
    token_type_table = dl_net.Variable(name=token_type_embedding_name,
                                       shape={'dim': [token_type_vocab_size, embedding_size]},
                                       initializer=CreateInitializer(initializer_range))
    token_type_embeddings = dl_net.Gather(token_type_table, token_type_ids_blob)
    output = dl_net.Add([output, token_type_embeddings])

  if use_position_embeddings:
    position_table = dl_net.Variable(name=position_embedding_name,
                                     shape={'dim': [1, max_position_embeddings, embedding_size]},
                                     initializer=CreateInitializer(initializer_range))
    assert seq_length <= max_position_embeddings
    if seq_length != max_position_embeddings:
      position_table = dl_net.Slice(position_table, dim_slice_conf=[{'end': seq_length}, {}])
    output = dl_net.BroadcastAdd(output, position_table)

  output = _LayerNorm(output, embedding_size)
  output = _Dropout(output, dropout_prob)

  return output


def _EmbeddingLookup(input_ids_blob,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings"):
  dl_net = input_ids_blob.dl_net()
  embedding_table = dl_net.Variable(name=word_embedding_name, shape={'dim':[vocab_size, embedding_size]},
                                    initializer=CreateInitializer(initializer_range))
  output = dl_net.Gather(embedding_table, input_ids_blob)
  return output, embedding_table

def GetActivation(name):
  if name == 'linear':
    return op_conf_util.kNone
  elif name == 'relu':
    return op_conf_util.kRelu
  elif name == 'tanh':
    return op_conf_util.kTanH
  elif name == 'gelu':
    return _Gelu
  else:
    raise Exception("unsupported activation")

