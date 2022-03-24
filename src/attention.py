#!/usr/bin/env python
# -*- coding: utf-8 -*-

class SimpleAttention(tf.keras.models.Model):

    '''
    Attention の説明をするための、 Multi-head ではない単純な Attention 
    '''
    def __init__(self, depth: int, *args, **kwargs):
        '''
        コンストラクタです :param depth: 隠れ層及び出力の次元
        '''
        super().__init__(*args, **kwargs)
        self.depth = depth

        self.q_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='v_dense_layer')
        self.output_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='output_dense_layer')

     def call(self, input: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        '''
        モデルの実行を行います。
        :param input: query のテンソル
        :param memory: query に情報を与える memory のテンソル
        '''
        q = self.q_dense_layer(input)  # [batch_size, q_length, depth]
        k = self.k_dense_layer(memory)  # [batch_size, m_length, depth]
        v = self.v_dense_layer(memory)

        # ここでqとkの内積を取ることで、query と key の関連度のようなものを計算
        logit = tf.matmul(q, k, transpose_b=True)  # [batch_size, q_length, k_length]

        # softmax を取ることで正規化
        attention_weight = tf.nn.softmax(logit, name='attention_weight')

        # 重みに従って value から情報を引く
        attention_output = tf.matmul(attention_weight, v)  # [batch_size, q_length, depth]
        return self.output_dense_layer(attention_output)

