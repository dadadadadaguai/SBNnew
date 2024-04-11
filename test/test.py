import argparse

import torch
from transformers import BertConfig, BertTokenizer
from transformers import BertModel
from torch import nn as nn
import torch.nn.functional as F
class SelfAttention(torch.nn.Module):
    def __init__(self, args):
        super(SelfAttention,self).__init__()
        self.args = args
        self.linear_q = torch.nn.Linear(args.lstm_dim * 2, args.lstm_dim * 2)
        self.w_query = torch.nn.Linear(args.lstm_dim * 2, args.lstm_dim)
        self.w_value = torch.nn.Linear(args.lstm_dim * 2, args.lstm_dim)
        self.v = torch.nn.Linear(args.lstm_dim, 1, bias=False)

    def forward(self, query, value, mask):
        attention_states = query
        attention_states_T = value
        attention_states_T = attention_states_T.permute([0, 2, 1])

        weights=torch.bmm(attention_states, attention_states_T)
        weights = weights.masked_fill(mask.unsqueeze(1).expand_as(weights)==0, float("-inf"))
        attention = F.softmax(weights,dim=2)

        merged=torch.bmm(attention, value)
        merged=merged * mask.unsqueeze(2).float().expand_as(merged)

        return merged

    def forward_perceptron(self, query, value, mask):
        attention_states = query
        attention_states = self.w_query(attention_states)
        attention_states = attention_states.unsqueeze(2).expand(-1,-1,attention_states.shape[1], -1)

        attention_states_T = value
        attention_states_T = self.w_value(attention_states_T)
        attention_states_T = attention_states_T.unsqueeze(2).expand(-1,-1,attention_states_T.shape[1], -1)
        attention_states_T = attention_states_T.permute([0, 2, 1, 3])

        weights = torch.tanh(attention_states+attention_states_T)
        weights = self.v(weights).squeeze(3)
        weights = weights.masked_fill(mask.unsqueeze(1).expand_as(weights)==0, float("-inf"))
        attention = F.softmax(weights,dim=2)

        merged = torch.bmm(attention, value)
        merged = merged * mask.unsqueeze(2).float().expand_as(merged)
        return merged


if __name__ == "__main__":
    device = 'cpu'

    Bert = BertModel.from_pretrained('../pretrained_models/bert-base-uncased')
    bert_config = Bert.config
    Bert.to(device)
    emb_dim = 768
    hidden_dim = 100
    layers = 2
    is_bidirectional = True
    drop_rate = 0.5
    batch_size = 16  # 批量大小，这里设置为 10
    input_length = 20  # 输入序列长度，这里设置为 20
    num_directions = 2 if is_bidirectional else 1  # 如果是双向的话则为2，否则为1
    hidden =torch.randn(layers * num_directions,1,hidden_dim)    # 初始化隐藏状态的形状

    tokens_tensor=torch.randn(16,100).to(device)
    attention_mask=torch.randn(16,100).to(device)
    context_masks = torch.ones(batch_size, input_length,emb_dim).to(device)  # 替换为你的输入数据

    lstm = nn.LSTM(emb_dim, hidden_dim, layers, batch_first=True,
                   bidirectional=is_bidirectional, dropout=drop_rate).to(device)

    arg = argparse.Namespace(lstm_dim=hidden_dim)  # 设置 SelfAttention 所需参数
    attention_layer = SelfAttention(arg)
    lstm_dropout = nn.Dropout(drop_rate)

    # 初始化 BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('../pretrained_models/bert-base-uncased')

    # 待测试的文本
    text = "The drinks are a saving grace , but service staff , please , get over yourselves."

    # 使用 tokenizer 对文本进行处理，得到 token ids 和 attention mask
    encoded_input = tokenizer(text, padding='max_length', truncation=True, max_length=20, return_tensors='pt')

    # 获取 token ids 和 attention mask
    tokens_tensor = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    # 确保类型为 LongTensor
    tokens_tensor = tokens_tensor.long().to(device)
    attention_mask = attention_mask.long().to(device)

    # 使用之前加载的预训练模型 Bert
    h = Bert(input_ids=tokens_tensor, attention_mask=attention_mask)[0]
    h.to(device)

    output, _ = lstm(h, hidden)
    bert_lstm_output = lstm_dropout(output)
    bert_lstm_att_feature = bert_lstm_output
    print(bert_lstm_att_feature.shape)




