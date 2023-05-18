# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
import numpy as np
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNet,SqueezeExcitation
import logging
from tqdm import tqdm
import os, sys
import base64
import hashlib
import multiprocessing




class DCN_MEMONETV3(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DCN_MEMONETV3",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 mid_dim = 20,
                 fields_num = 19,
                 use_kif = False,
                 n_rows = 500000,
                 dnn_hidden_units=[1000,1000],
                 num_cross_layers = 3,
                 dnn_activations="ReLU",
                 net_dropout=0.4,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DCN_MEMONETV3, self).__init__(feature_map,
                                  model_id=model_id,
                                  gpu=gpu,
                                  embedding_regularizer=embedding_regularizer,
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.device = torch.cuda.current_device()

        #one_order embedding
        self.embedding_dim = embedding_dim
        self.mid_layer_dim = mid_dim

        #codebook
        self.n_rows = n_rows
        #与one-order embedding维度保持一致
        self.l_columns = embedding_dim
        #two-order
        self.codebook = nn.Embedding(self.n_rows,self.l_columns)
        '''
        class YourModel:
    def __init__(self, n_rows, l_columns):
        self.n_rows = n_rows
        self.l_columns = l_columns
        self.codebook = tf.Variable(tf.random.normal([n_rows, l_columns]))

    def get_embedding(self, indices):
        embeddings = tf.nn.embedding_lookup(self.codebook, indices)
        return embeddings
        
        '''
        #three-order
        self.codebookV3 = nn.Embedding(self.n_rows,self.l_columns)

        self.m_hash_num = 2

        self.hidden_units = dnn_hidden_units
        self.output_activation = self.get_output_activation('binary_classification')

        #TODO  特征数量依数据集而变化
        #KKBox_x1  19:  tiny_csv:18
        self.field_num =  fields_num
        self.use_kif = use_kif
        self.key_index = [1,14,17,18]
        self.key_fields_num = len(self.key_index)
        self.three_order_cross_num = int((self.key_fields_num-1)*(self.key_fields_num-2)/2)


        #one-order + two-order + three-order
        self.dnn_input_dim = (self.field_num+2*self.key_fields_num)*self.embedding_dim

        #TODO two-order HCNET
        #AMR component
        self.attentive_layer = nn.Sequential(
            nn.Linear(2*self.embedding_dim,self.mid_layer_dim,bias=False),
            nn.ReLU(),
            nn.Linear(self.mid_layer_dim,self.m_hash_num*self.l_columns,bias=False),
            nn.Identity()
        )

        #TF
        '''
        self.attentive_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(self.mid_layer_dim, input_shape=(2 * self.embedding_dim,), use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.m_hash_num * self.l_columns, use_bias=False),
        ])
        '''


        #AMR component project cij to a new embedding space
        self.amr_mlp_layer = nn.Sequential(
            nn.Linear(self.m_hash_num*self.l_columns,self.embedding_dim,bias=False),
            nn.Identity()
        )

        '''
        self.amr_mlp_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(self.embedding_dim, input_shape=(self.m_hash_num*self.l_columns,), use_bias=False),
        ])
        '''


        self.gas_attentive_layer = nn.Sequential(
            nn.Linear(self.key_fields_num * self.embedding_dim, self.mid_layer_dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.mid_layer_dim, self.key_fields_num * (self.key_fields_num - 1), bias=False),
            nn.Identity()
        )

        '''
        self.gas_attentive_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(self.mid_layer_dim, input_shape=(self.key_fields_num * self.embedding_dim,), use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.key_fields_num * (self.key_fields_num - 1), use_bias=False),
        ])
        '''

        #TODO three-order HCNET

        self.gas_attentive_layer_V3 = nn.Sequential(
            nn.Linear(self.key_fields_num * self.embedding_dim, self.mid_layer_dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.mid_layer_dim, self.key_fields_num*self.three_order_cross_num, bias=False),
            nn.Identity()
        )

        self.attentive_layer_V3 = nn.Sequential(
            nn.Linear(3 * self.embedding_dim, self.mid_layer_dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.mid_layer_dim, self.m_hash_num * self.l_columns, bias=False),
            nn.Identity()
        )

        self.amr_mlp_layer_V3 = nn.Sequential(
            nn.Linear(self.m_hash_num * self.l_columns, self.embedding_dim, bias=False),
            nn.Identity()
        )




        #DCN part
        self.dnn = MLP_Block(input_dim=self.dnn_input_dim,
                             output_dim=None,  # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.crossnet = CrossNet(self.dnn_input_dim, num_cross_layers)
        self.final_dim = self.dnn_input_dim + dnn_hidden_units[-1]
        self.fc = nn.Linear(self.final_dim, 1)  # [cross_part, dnn_part] -> logit



        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self,inputs):
        #分离feature和label
        X,raw_inputs = self.get_inputs(inputs)

        one_order_emb = self.embedding_layer(X)
        two_order_emb = self.HCNet_batch_KIF(raw_inputs,one_order_emb)
        #
        three_order_emb = self.HCNetV3(raw_inputs,one_order_emb)

        one_order_flat = one_order_emb.view(one_order_emb.size(0), -1)

        dcn_input = torch.cat([one_order_flat,two_order_emb,three_order_emb],dim=1)

        #DCN part
        dnn_output = self.dnn(dcn_input)
        cross_output = self.crossnet(dcn_input)
        #Final part
        final_out = torch.cat([cross_output, dnn_output], dim=-1)
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}

        return return_dict


    def get_aij_batch(self,batch_xij):
        result_list = []
        for feature_xij in batch_xij:
            temp_aij_list = self.hash_func(feature_xij)
            result_list.append(temp_aij_list)
        result_list = np.array(result_list)
        return result_list
    '''
    def get_aij_batch(self, batch_xij):
        result_list = []
        for feature_xij in batch_xij:
            temp_aij_list = self.hash_func(feature_xij)
            result_list.append(temp_aij_list)
        result_list = np.array(result_list)
        return result_list
    '''

    def get_cij_batch(self,batch_index):

        index_list = torch.from_numpy(batch_index).long().to(self.device) # 转换为 LongTensor

        # 将 index_list 中的第一列作为索引，取 codebook 中对应的行向量
        output1 = torch.index_select(self.codebook.weight, 0, index_list[:, 0])

        # 将 index_list 中的第二列作为索引，取 codebook 中对应的行向量
        output2 = torch.index_select(self.codebook.weight, 0, index_list[:, 1])

        # 将 output1 和 output2 进行拼接，得到 [batchSize, 2, l] 的输出
        output = torch.stack([output1, output2], dim=1)
        return output

    '''
    def get_cij_batch(self, batch_index):
        index_list = tf.constant(batch_index, dtype=tf.int32)
        codebook = tf.Variable(...)  # Define your codebook tensor

        # 将 index_list 中的第一列作为索引，取 codebook 中对应的行向量
        output1 = tf.gather(codebook, index_list[:, 0])

        # 将 index_list 中的第二列作为索引，取 codebook 中对应的行向量
        output2 = tf.gather(codebook, index_list[:, 1])

        # 将 output1 和 output2 进行拼接，得到 [batchSize, 2, l] 的输出
        output = tf.stack([output1, output2], axis=1)
        return output
    '''

    def get_cij_batch_V3(self, batch_index):
        index_list = torch.from_numpy(batch_index).long().to(self.device)  # 转换为 LongTensor
        output1 = torch.index_select(self.codebookV3.weight, 0, index_list[:, 0])
        output2 = torch.index_select(self.codebookV3.weight, 0, index_list[:, 1])
        output = torch.stack([output1, output2], dim=1)
        return output



    #input_str:数字字符串
    #MD5,SHA-1
    def hash_func(self,input_str):
        #MD5
        md5_out = hashlib.md5(input_str.encode()).hexdigest()
        #SHA-1
        sha1_out = hashlib.sha1(input_str.encode()).hexdigest()

        sha1_int = int(sha1_out, 16)
        md5_int = int(md5_out, 16)

        md5_mode = md5_int % self.n_rows
        sha1_mod = sha1_int % self.n_rows

        return [md5_mode,sha1_mod]

    #cij_tensor:  [batch_size,2,l]
    def attentive_memory_restoring_batch(self,emb_i,emb_j,cij_tensor):
        #[batch,2d]
        input_Z = torch.cat([emb_i, emb_j], dim=1)

        #[batch,2l]
        mask_I = self.attentive_layer(input_Z)

        #change shape to [batch,2,l]
        mask_Ir = torch.reshape(mask_I,(-1,self.m_hash_num,self.l_columns))

        #[batch,2,l]
        result = torch.mul(mask_Ir,cij_tensor)

        #[batch,m*l]
        result_flatten = torch.flatten(result,start_dim=1)

        #[batch,d]
        vij = self.amr_mlp_layer(result_flatten)
        return vij

    '''
    def attentive_memory_restoring_batch(self, emb_i, emb_j, cij_tensor):
        # [batch, 2d]
        input_Z = tf.concat([emb_i, emb_j], axis=1)

        # [batch, 2l]
        mask_I = self.attentive_layer(input_Z)

        # Change shape to [batch, 2, l]
        mask_Ir = tf.reshape(mask_I, (-1, self.m_hash_num, self.l_columns))

        # [batch, 2, l]
        result = tf.multiply(mask_Ir, cij_tensor)

        # [batch, m*l]
        result_flatten = tf.reshape(result, (-1, self.m_hash_num * self.l_columns))

        # [batch, d]
        vij = self.amr_mlp_layer(result_flatten)
        return vij
    '''

    def attentive_memory_restoring_batch_V3(self,emb_i,emb_j,emb_k,cijk_tensor):
        input_Z = torch.cat([emb_i,emb_j,emb_k],dim=1)
        mask_I = self.attentive_layer_V3(input_Z)
        # change shape to [batch,2,l]
        mask_Ir = torch.reshape(mask_I, (-1, self.m_hash_num, self.l_columns))

        # [batch,2,l]
        result = torch.mul(mask_Ir, cijk_tensor)

        # [batch,m*l]
        result_flatten = torch.flatten(result, start_dim=1)

        # [batch,d]
        vijk = self.amr_mlp_layer_V3(result_flatten)

        return vijk

    #batch_R [batch, f, (f - 1) * (f - 2) / 2]




    def feature_shrinking_batch(self,index_i,vi_list,weight_matrix):
        #weight_matrix: [batch,f-1]
        matrix_a_batch = weight_matrix[:,index_i,:]

        #[batch,f-1,d]
        vi_matrix = torch.stack(vi_list, dim=1)

        #[batch,f-1,d]
        result = matrix_a_batch.unsqueeze(-1) * vi_matrix
        #[batch,d]
        result_sum = torch.sum(result, dim=1)

        return result_sum
    '''
    def feature_shrinking_batch(self, index_i, vi_list, weight_matrix):
        # weight_matrix: [batch,f-1]
        matrix_a_batch = tf.gather(weight_matrix, index_i, axis=1)

        # [batch,f-1,d]
        vi_matrix = tf.stack(vi_list, axis=1)

        # [batch,f-1,d]
        result = tf.expand_dims(matrix_a_batch, -1) * vi_matrix
        # [batch,d]
        result_sum = tf.reduce_sum(result, axis=1)

        return result_sum
    '''


    def HCNet_batch(self,inputs,one_order_emb):
        #inputs: [batch_size,fields_num]
        batch_emb_flat = one_order_emb.reshape(one_order_emb.shape[0],-1)

        batch_R_flat = self.gas_attentive_layer(batch_emb_flat)

        #[batch,f,f-1]
        if self.use_kif:
            batch_R = batch_R_flat.view(-1,self.key_fields_num,self.key_fields_num-1)
        else:
            batch_R = batch_R_flat.view(-1,self.field_num,self.field_num-1)

        #编码所有样本
        encode_inputs = self.encode_input_batch(inputs)

        #转为numpy.ndarray
        encode_inputs = np.array(encode_inputs)
        batch_v2_list = []
        #依次遍历特征
        for i  in range(inputs.shape[1]):
            batch_f_i = encode_inputs[:,i]
            #print("batch_f_i[:5] : ",batch_f_i[:5])

            #[batch_size, emb_dim]
            batch_v_i = one_order_emb[:,i]
            E_list = []
            #i>= j v_ij =  v_ji
            for j in range(inputs.shape[1]):
                if i == j:
                    continue
                else:
                    batch_f_j = encode_inputs[:,j]
                    batch_v_j = one_order_emb[:,j]
                    #f_ij = f_ji,保证下标更小的在前
                    if i < j :
                        #ndarray  [batch_size] ['02000030300004' '02000070300008' '02000110300012']
                        batch_f_ij = np.core.defchararray.add(batch_f_i, batch_f_j)
                    else:
                        batch_f_ij = np.core.defchararray.add(batch_f_j, batch_f_i)

                    #TODO
                    #ndarray  [batch_size, 2]
                    batch_aij_list = self.get_aij_batch(batch_f_ij)

                    #tensor [batch_size, 2, l]
                    batch_cij = self.get_cij_batch(batch_aij_list)

                    #batch_v_ij: [batch,d]
                    batch_v_ij = self.attentive_memory_restoring_batch(batch_v_i, batch_v_j, batch_cij)


                    E_list.append(batch_v_ij)

            #[batch,d]
            batch_v_i_2 = self.feature_shrinking_batch(i, E_list, batch_R)
            batch_v2_list.append(batch_v_i_2)
        #遍历完所有的特征,把v2_list按列拼接，再转为[batch_size,f*d]的tensor
        #[batch,f,d]
        two_order_emb = torch.stack(batch_v2_list,dim=1)
        two_order_emb_flat = torch.flatten(two_order_emb,start_dim=1)
        return two_order_emb_flat

    '''
    def HCNet_batch(self, inputs, one_order_emb):
        # inputs: [batch_size,fields_num]
        batch_emb_flat = tf.reshape(one_order_emb, [tf.shape(one_order_emb)[0], -1])

        batch_R_flat = self.gas_attentive_layer(batch_emb_flat)

        # [batch,f,f-1]
        if self.use_kif:
            batch_R = tf.reshape(batch_R_flat, [-1, self.key_fields_num, self.key_fields_num-1])
        else:
            batch_R = tf.reshape(batch_R_flat, [-1, self.field_num, self.field_num-1])

        # 编码所有样本
        encode_inputs = self.encode_input_batch(inputs)

        # 转为numpy.ndarray
        encode_inputs = np.array(encode_inputs)
        batch_v2_list = []
        # 依次遍历特征
        for i in range(inputs.shape[1]):
            batch_f_i = encode_inputs[:, i]

            # [batch_size, emb_dim]
            batch_v_i = one_order_emb[:, i]
            E_list = []
            # i >= j v_ij =  v_ji
            for j in range(inputs.shape[1]):
                if i == j:
                    continue
                else:
                    batch_f_j = encode_inputs[:, j]
                    batch_v_j = one_order_emb[:, j]
                    # f_ij = f_ji, 保证下标更小的在前
                    if i < j:
                        # ndarray  [batch_size]
                        batch_f_ij = np.core.defchararray.add(batch_f_i, batch_f_j)
                    else:
                        batch_f_ij = np.core.defchararray.add(batch_f_j, batch_f_i)

                    # TODO
                    # ndarray  [batch_size, 2]
                    batch_aij_list = self.get_aij_batch(batch_f_ij)

                    # tensor [batch_size, 2, l]
                    batch_cij = self.get_cij_batch(batch_aij_list)

                    # batch_v_ij: [batch,d]
                    batch_v_ij = self.attentive_memory_restoring_batch(batch_v_i, batch_v_j, batch_cij)

                    E_list.append(batch_v_ij)

            # [batch,d]
            batch_v_i_2 = self.feature_shrinking_batch(i, E_list, batch_R)
            batch_v2_list.append(batch_v_i_2)
        # 遍历完所有的特征,把v2_list按列拼接，再转为[batch_size,f*d]的tensor
        # [batch,f,d]
        two_order_emb = tf.stack(batch_v2_list, axis=1)
        two_order_emb_flat = tf.reshape(two_order_emb, [tf.shape(two_order_emb)[0], -1])
        return two_order_emb_flat
    '''

    #主体是HCNet_batch,仅保留数量最多的前四个fields,选取inputs及one_order_emb中特定的列
    def HCNet_batch_KIF(self, inputs, one_order_emb):
        # inputs: [batch_size,fields_num]
        # one_order_emb: [batch_size, fields_num, embedding_dim]
        inputs_kif,one_order_kif = self.select_KIF(inputs,one_order_emb)
        two_order_emb  = self.HCNet_batch(inputs_kif,one_order_kif)
        return two_order_emb

    def select_KIF(self, inputs, one_order_emb):
        # ndarray
        selected_inputs = inputs[:, self.key_index]
        # tensor
        selected_emb = one_order_emb[:, self.key_index, :]
        return selected_inputs, selected_emb

    '''
    def HCNet_batch_KIF(self, inputs, one_order_emb):
        # inputs: [batch_size,fields_num]
        # one_order_emb: [batch_size, fields_num, embedding_dim]
        inputs_kif, one_order_kif = self.select_KIF(inputs, one_order_emb)
        two_order_emb = self.HCNet_batch(inputs_kif, one_order_kif)
        return two_order_emb

    def select_KIF(self, inputs, one_order_emb):
        # ndarray
        selected_inputs = inputs[:, self.key_index]
        # tensor
        selected_emb = tf.gather(one_order_emb, self.key_index, axis=1)
        return selected_inputs, selected_emb
    '''



    def HCNetV3(self, inputs, one_order_emb):
        inputs_kif, one_order_kif = self.select_KIF(inputs, one_order_emb)

        # inputs: [batch_size,fields_num]
        batch_emb_flat = one_order_kif.reshape(one_order_kif.shape[0], -1)

        #[batch_size,f,(f-1)*(f-2)/2]
        batch_R_flat = self.gas_attentive_layer_V3(batch_emb_flat)

        # [batch,f,(f-1)*(f-2)/2]
        batch_R = batch_R_flat.view(-1, self.key_fields_num, self.three_order_cross_num )

        # 编码所有样本
        encode_inputs = self.encode_input_batch(inputs_kif)

        # 转为numpy.ndarray
        encode_inputs = np.array(encode_inputs)
        batch_v3_list = []
        # 依次遍历特征
        for i in range(self.key_fields_num):
            batch_f_i = encode_inputs[:, i]
            #[batch_size, emb_dim]
            batch_v_i = one_order_emb[:, i]

            E_list = []
            # i>= j v_ij =  v_ji
            for j in range(inputs_kif.shape[1]):
                if j != i:
                    batch_f_j = encode_inputs[:, j]
                    batch_v_j = one_order_emb[:, j]
                    for k in range(inputs_kif.shape[1]):
                        if k!=i and k!= j:
                            batch_f_k = encode_inputs[:, k]
                            batch_v_k = one_order_emb[:, k]
                            # f_ijk = f_kji,保证下标更小的在前
                            if i < j  < k:
                                # ndarray  [batch_size] ['02000030300004' '02000070300008' '02000110300012']
                                batch_f_ijk = self.np_str_add(batch_f_i, batch_f_j,batch_f_k)

                            elif i < k < j:
                                batch_f_ijk = self.np_str_add(batch_f_i, batch_f_k,batch_f_j)
                            elif j < i <k:
                                batch_f_ijk = self.np_str_add(batch_f_j, batch_f_i,batch_f_k)
                            elif j < k < i:
                                batch_f_ijk = self.np_str_add(batch_f_j, batch_f_k,batch_f_i)
                            elif k < i <j:
                                batch_f_ijk = self.np_str_add(batch_f_k, batch_f_i,batch_f_j)
                            elif k < j < i:
                                batch_f_ijk = self.np_str_add(batch_f_k, batch_f_j,batch_f_i)
                            else:
                                print("batch_f_ijk wrong!")

                    #TODO 和2阶区别仅仅在于输入的字符串变长
                    # ndarray  [batch_size, 2]
                    batch_aijk_list = self.get_aij_batch(batch_f_ijk)

                    #TODO tensor [batch_size, 2, l]  和2阶区别在于从不同的codebook中取值
                    batch_cijk = self.get_cij_batch_V3(batch_aijk_list)

                    # batch_v_ijk: [batch,d]
                    batch_v_ijk = self.attentive_memory_restoring_batch_V3(batch_v_i, batch_v_j,batch_v_k,batch_cijk)
                    E_list.append(batch_v_ijk)

            # [batch,d]  TODO  和 shriking部分和2阶一致
            batch_v_i_3 = self.feature_shrinking_batch(i, E_list, batch_R)
            batch_v3_list.append(batch_v_i_3)
        # 遍历完所有的特征,把v3_list按列拼接，再转为[batch_size,f*d]的tensor
        # [batch,f,d]
        three_order_emb = torch.stack(batch_v3_list, dim=1)
        three_order_emb_flat = torch.flatten(three_order_emb, start_dim=1)
        return three_order_emb_flat

    def np_str_add(self,i,j,k):
        batch_ij = np.core.defchararray.add(i,j)
        batch_ijk = np.core.defchararray.add(batch_ij, k)
        return batch_ijk






    #如果id大于99999则只保留后5位，否则保留原值
    def encode_input(self,raw_inputs):
        temp_inputs = torch.where(torch.from_numpy(raw_inputs) > torch.tensor(99999), torch.from_numpy(raw_inputs) %  torch.tensor(100000), torch.from_numpy(raw_inputs))
        input_str = temp_inputs.numpy().astype(str)
        inputs_filled = [elem.zfill(5) for elem in input_str]
        encode_inputs = [f'{i:02d}{elem}' for i, elem in enumerate(inputs_filled)]
        return encode_inputs

    def encode_input_batch(self,inputs):
        encode_list = []
        for i in range(inputs.shape[0]):
            encode_out = self.encode_input(inputs[i])
            encode_list.append(encode_out)
        return encode_list

    '''
    def encode_input(self, raw_inputs):
        temp_inputs = np.where(raw_inputs > 99999, raw_inputs % 100000, raw_inputs)
        input_str = temp_inputs.astype(str)
        inputs_filled = [elem.zfill(5) for elem in input_str]
        encode_inputs = [f'{i:02d}{elem}' for i, elem in enumerate(inputs_filled)]
        return encode_inputs

    def encode_input_batch(self, inputs):
        encode_list = []
        for i in range(inputs.shape[0]):
            encode_out = self.encode_input(inputs[i])
            encode_list.append(encode_out)
        return encode_list
    '''


    def get_inputs(self, inputs, feature_source=None):
        if feature_source and type(feature_source) == str:
            feature_source = [feature_source]
        X_dict = dict()
        input_dict = dict()
        #index = 0
        for feature, spec in self.feature_map.features.items():
            if (feature_source is not None) and (spec["source"] not in feature_source):
                continue
            if spec["type"] == "meta":
                continue
            #print("index: ",index,"  feature name: ",feature)
            #index += 1
            #[batch_size]
            input_dict[feature] = inputs[:, self.feature_map.get_column_index(feature)]
            X_dict[feature] = inputs[:, self.feature_map.get_column_index(feature)].to(self.device)
            #print("X_dict[feature] shape: X_dict[feature]",X_dict[feature].shape)

        X_list = [input_dict[key].reshape(-1, 1) for key in input_dict]
        X_concat = np.concatenate(X_list, axis=1)

        return X_dict,X_concat




