import bert as bert_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util

def PreTrain(input_ids_blob,
             input_mask_blob,
             token_type_ids_blob,
             masked_lm_positions_blob,
             masked_lm_ids_blob,
             masked_lm_weights_blob,
             next_sentence_label_blob,
             vocab_size,
             seq_length=512,
             hidden_size=768,
             num_hidden_layers=12,
             num_attention_heads=12,
             intermediate_size=3072,
             hidden_act='gelu',
             hidden_dropout_prob=0.1,
             attention_probs_dropout_prob=0.1,
             max_position_embeddings=512,
             type_vocab_size=16,
             max_predictions_per_seq=20,
             initializer_range=0.02):
  backbone = bert_util.BertBackbone(
      input_ids_blob=input_ids_blob,
      input_mask_blob=input_mask_blob,
      token_type_ids_blob=token_type_ids_blob,
      vocab_size=vocab_size,
      seq_length=seq_length,
      hidden_size=hidden_size,
      num_hidden_layers=num_hidden_layers,
      num_attention_heads=num_attention_heads,
      intermediate_size=intermediate_size,
      hidden_act=hidden_act,
      hidden_dropout_prob=hidden_dropout_prob,
      attention_probs_dropout_prob=attention_probs_dropout_prob,
      max_position_embeddings=max_position_embeddings,
      type_vocab_size=type_vocab_size,
      initializer_range=initializer_range)

  (lm_loss, _, _) = _AddMaskedLanguageModelLoss(
      input_blob=backbone.sequence_output(),
      output_weights_blob=backbone.embedding_table(),
      positions_blob=masked_lm_positions_blob,
      label_id_blob=masked_lm_ids_blob,
      label_weight_blob=masked_lm_weights_blob,
      seq_length=seq_length,
      hidden_size=hidden_size,
      vocab_size=vocab_size,
      max_predictions_per_seq=max_predictions_per_seq,
      hidden_act=bert_util.GetActivation(hidden_act),
      initializer_range=initializer_range)
  pooled_output = PooledOutput(backbone.sequence_output(), hidden_size, initializer_range)
  (ns_loss, _, _) = _AddNextSentenceOutput(
      input_blob=pooled_output,
      label_blob=next_sentence_label_blob,
      hidden_size=hidden_size,
      initializer_range=initializer_range)
  dl_net = input_ids_blob.dl_net()
  with dl_net.VariableScope("cls-loss"):
    total_loss = dl_net.BroadcastAdd(lm_loss, ns_loss)
  return dl_net.IdentityLoss(total_loss, name='identity_loss')
  #return loss

def PooledOutput(sequence_output, hidden_size, initializer_range):
  dl_net = sequence_output.dl_net()
  with dl_net.VariableScope("bert-pooler"):
    first_token_tensor = dl_net.Slice(sequence_output, dim_slice_conf=[{'start': 0, 'end': 1}, {}])
    first_token_tensor = dl_net.Reshape( first_token_tensor, shape={'dim': [hidden_size]})
    pooled_output = bert_util._FullyConnected(
        first_token_tensor,
        input_size=hidden_size,
        units=hidden_size,
        weight_initializer=bert_util.CreateInitializer(initializer_range),
        name='dense')
    pooled_output = dl_net.TanH(pooled_output)
  return pooled_output

def _AddMaskedLanguageModelLoss(input_blob,
                                output_weights_blob,
                                positions_blob,
                                label_id_blob,
                                label_weight_blob,
                                seq_length,
                                hidden_size,
                                vocab_size,
                                max_predictions_per_seq,
                                hidden_act,
                                initializer_range):
  dl_net = input_blob.dl_net()

  with dl_net.VariableScope("other"):
    sum_label_weight_blob = dl_net.ReduceSum(label_weight_blob, axis=[-1])
    zeros = dl_net.ScalarMul(sum_label_weight_blob, 0.0)
    ones = dl_net.ScalarAdd(zeros, 1.0)
    sum_label_weight_blob = dl_net.ReduceSum(sum_label_weight_blob)
    batch_size = dl_net.ReduceSum(ones)
    sum_label_weight_blob = dl_net.BroadcastDiv(sum_label_weight_blob, batch_size)
  with dl_net.VariableScope("cls-predictions"):
    input_blob = _GatherIndexes(input_blob, positions_blob, seq_length, hidden_size)
    with dl_net.VariableScope("transform"):
      if callable(hidden_act):
        act_fn = op_conf_util.kNone
      else:
        act_fn = hidden_act
      input_blob = bert_util._FullyConnected(
        input_blob,
        input_size=hidden_size,
        units=hidden_size,
        activation=act_fn,
        weight_initializer=bert_util.CreateInitializer(initializer_range),
        name='dense')
      if callable(hidden_act):
        input_blob = hidden_act(input_blob)
        input_blob = bert_util._LayerNorm(input_blob, hidden_size)
    output_bias = dl_net.Variable(name="output_bias", shape={'dim': [vocab_size]},
                                  initializer={'constant_conf': {'value': 1}}, model_name='bias')
    logit_blob = dl_net.Matmul(input_blob, output_weights_blob, transpose_b=True)
    logit_blob = dl_net.BiasAdd(logit_blob, output_bias)
    logit_blob = dl_net.Softmax(logit_blob)
    label_id_blob = bert_util.Reshape(label_id_blob, [-1])
    pre_example_loss = dl_net.SparseCrossEntropy(prediction=logit_blob, label=label_id_blob)
    numerator = dl_net.BroadcastMul(bert_util.Reshape(pre_example_loss,
                                    [-1, max_predictions_per_seq]), label_weight_blob)
    with dl_net.VariableScope("loss"):
      numerator = dl_net.ReduceSum(numerator, axis=[-1])
      denominator = dl_net.ScalarAdd(sum_label_weight_blob,1e-5)
      loss = dl_net.BroadcastDiv(numerator, denominator)
    return loss, pre_example_loss, logit_blob


def _GatherIndexes(sequence_blob, positions_blob, seq_length, hidden_size):
  dl_net = sequence_blob.dl_net()
  output = dl_net.BatchGather(in_blob=sequence_blob, indices_blob=positions_blob)
  output = bert_util.Reshape(output, [-1, hidden_size])
  return output


def _AddNextSentenceOutput(input_blob, label_blob, hidden_size, initializer_range):
  dl_net = input_blob.dl_net()
  with dl_net.VariableScope("cls-seq_relationship"):
    output_weight_blob = dl_net.Variable(
        name="output_weights",
        shape={'dim': [2, hidden_size]},
        initializer=bert_util.CreateInitializer(initializer_range))
    output_bias_blob = dl_net.Variable(
        name="output_bias",
        shape={'dim': [2]},
        initializer={'constant_conf': {'value': 0.0}}, model_name='bias')
    logit_blob = dl_net.Matmul(input_blob, output_weight_blob, transpose_b=True)
    logit_blob = dl_net.BiasAdd(logit_blob, output_bias_blob)
    logit_blob = dl_net.Softmax(logit_blob)
    pre_example_loss = dl_net.SparseCrossEntropy(prediction=logit_blob, label=label_blob)
    loss = pre_example_loss
    #with dl_net.VariableScope("loss"):
    #  loss = dl_net.ReduceSum(pre_example_loss, axis=[-1])
    return loss, pre_example_loss, logit_blob
