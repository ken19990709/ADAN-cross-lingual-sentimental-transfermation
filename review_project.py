import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import spacy
from layers import *
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt


BATCH_SIZE = 32
EMB_DIM = 300
HID_DIM = 300
BID = True
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        inputs, lengths = input
        batch_size, max_len, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = functional.softmax(logits, dim=-1)

        # computing mask
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0).to(inputs.device)
        mask = (idxes<lengths.unsqueeze(1)).float()

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)
        return output


class embedding(nn.Module):
    def __init__(self, vocab_size, embedding_tensor, emb_dim):
        super(embedding, self).__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.embedding_layer = nn.Embedding(self.vocab_size, self.emb_dim, _weight=embedding_tensor)
        # nn.init.eye(self.map1.weight)

    def forward(self, x):
        out = self.embedding_layer(x)
        return out


class Feature_extractor(nn.Module):  # input [batch_size,length,embedding]
    def __init__(self, emb_dim, hid_dim, n_layers=2, dropout=0.2, bdrnn=True):
        super(Feature_extractor, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bdrnn = bdrnn
        self.dropout = dropout
        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, num_layers=self.n_layers, batch_first=True
                           , bidirectional=bdrnn, dropout=dropout)
        self.attn = DotAttentionLayer(self.hid_dim*2) if self.bdrnn else DotAttentionLayer(self.hid_dim)

    def forward(self, embedded,lengths):
        lengths_list = lengths.tolist()
        embedded = pack_padded_sequence(embedded, lengths_list, batch_first=True,enforce_sorted=False)
        output, (ht, ct) = self.rnn(embedded)
        output = pad_packed_sequence(output, batch_first=True)[0]



        #return torch.sum(output, 1) / lengths.long().view(-1, 1)#average attention
        output = self.attn((output, lengths))
        return output
        pass


class SentimentClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(SentimentClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
        # self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        return self.net(input)


class LanguageDetector(nn.Module):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 dropout,
                 batch_norm=False):
        super(LanguageDetector, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.net.add_module('q-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('q-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('q-linear-final', nn.Linear(hidden_size, 2))

    def forward(self, input):
        return self.net(input)


def get_batch_data_fast(batch_size, data_en, data_cn, label_en,len_en,len_cn):
    random_en_indices = torch.LongTensor(batch_size).random_(data_en.size(0))
    random_cn_indices = torch.LongTensor(batch_size).random_(data_cn.size(0))
    en_batch = data_en[random_en_indices]
    en_label = label_en[random_en_indices]
    cn_batch = data_cn[random_cn_indices]
    en_len = len_en[random_en_indices]
    cn_len = len_cn[random_cn_indices]

    return en_batch, cn_batch, en_label, en_len, cn_len


def train():
    with open("pkl/en_reviews_embedding.pkl", "rb") as f:
        en_embedding = pickle.load(f)
    with open("pkl/en_reviews.pkl", "rb") as f:
        en_set = pickle.load(f)
    en_data = torch.tensor(en_set["data"]).long()
    en_target = torch.tensor(en_set["target"]).long()
    en_length=torch.tensor(en_set["len"]).long()
    en_lang = en_set["lang"]
    with open("pkl/cn_reviews_embedding.pkl", "rb") as f:
        ch_embedding = pickle.load(f)
    with open("pkl/cn_reviews.pkl", "rb") as f:
        ch_set = pickle.load(f)
    ch_data = torch.tensor(ch_set["data"]).long()
    ch_target = torch.tensor(ch_set["target"]).long()
    ch_lang = ch_set["lang"]
    ch_length = torch.tensor(ch_set["len"]).long()
    #ch_test, _, ch_target = get_batch_data_fast(1000, ch_data, en_data, ch_target)  # 取一个大小为1000的有标签数据集

    en_embedding = embedding(en_lang.n_words, en_embedding, EMB_DIM).to(device)
    ch_embedding = embedding(ch_lang.n_words, ch_embedding, EMB_DIM).to(device)
    F = Feature_extractor(EMB_DIM, HID_DIM, 2, 0.2, BID)
    if BID:
        P = SentimentClassifier(2, HID_DIM * 2, 2, 0.2)
        Q = LanguageDetector(2, HID_DIM * 2, 0.2)
    else:
        P = SentimentClassifier(2, HID_DIM, 1, 0.2)
        Q = LanguageDetector(2, HID_DIM, 0.2)

    F, P, Q = F.to(device), P.to(device), Q.to(device)

    #F.load_state_dict(torch.load("model/netF_epoch_19.pth"))
    #P.load_state_dict(torch.load("model/netP_epoch_19.pth"))
    #Q.load_state_dict(torch.load("model/netQ_epoch_19.pth"))

    optimizer = optim.Adam(list(F.parameters()) + list(P.parameters()),
                           lr=0.0005)
    optimizerQ = optim.Adam(Q.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    num_epoch = 10
    round = 10000

    D_losses = []
    G_losses = []
    for epoch in range(0,num_epoch):
        print("epoch:", epoch)
        total_D_loss = 0
        total_G_loss = 0
        for i in range(0, round, BATCH_SIZE):


            en_batch, ch_batch, en_label,en_batch_len,ch_batch_len = get_batch_data_fast(BATCH_SIZE, en_data, ch_data, en_target,en_length,ch_length)
            en_batch, ch_batch, en_label,en_batch_len,ch_batch_len = en_batch.to(device)\
                , ch_batch.to(device), en_label.to(device),en_batch_len.to(device),ch_batch_len.to(device)

            adv_en_labels = torch.ones(BATCH_SIZE).long().to(device)
            adv_cn_labels = torch.zeros(BATCH_SIZE).long().to(device)

            en_batch = en_embedding(en_batch)
            ch_batch = ch_embedding(ch_batch)
            # D learning
            for D_iter in range(2):
                optimizerQ.zero_grad()
                en_F = F.forward(en_batch,en_batch_len)
                ch_F = F.forward(ch_batch,ch_batch_len)
                loss = criterion(Q(en_F).squeeze(1), adv_en_labels) + criterion(Q(ch_F).squeeze(1), adv_cn_labels)

                loss.backward(retain_graph=True)

                D_losses.append(loss.cpu().item())
                total_D_loss += loss.cpu().item()
                optimizerQ.step()

            # D learning

            # G learning
            for G_iter in range(1):
                optimizer.zero_grad()
                en_F = F.forward(en_batch,en_batch_len)
                ch_F = F.forward(ch_batch,ch_batch_len)

                adv_loss = criterion(Q(en_F).squeeze(1), adv_cn_labels) + criterion(Q(ch_F).squeeze(1), adv_en_labels)
                en_sentiment_prediction = P(en_F)

                sentiment_loss = criterion(en_sentiment_prediction.squeeze(1), en_label)

                g_loss = sentiment_loss + adv_loss  # 更注重情感分类的loss
                g_loss.backward(retain_graph=True)
                G_losses.append(g_loss.cpu().item())
                total_G_loss += g_loss.cpu().item()
                optimizer.step()

            if i % (10 * BATCH_SIZE) == 0 and i != 0:
                en_sentiment_label = en_sentiment_prediction.argmax(axis=1)
                acc_sen = (en_sentiment_label == en_label).sum() / BATCH_SIZE
                acc_d = (Q(en_F).argmax(axis=1) == adv_en_labels).sum() + (
                            Q(ch_F).argmax(axis=1) == adv_cn_labels).sum()
                acc_d = acc_d / (2 * BATCH_SIZE)
                print("avg G loss:%f Accuracy of sentiment detection:%f" % (total_G_loss / (i), acc_sen))
                print("avg D loss:%f Accuracy of D:%f " % (total_D_loss / (i), acc_d))
                print("------------------------")

            # G learning

        torch.save(F.state_dict(),
                   '{}/netF_epoch_{}.pth'.format("review_model", epoch))
        torch.save(P.state_dict(),
                   '{}/netP_epoch_{}.pth'.format("review_model", epoch))
        torch.save(Q.state_dict(),
                   '{}/netQ_epoch_{}.pth'.format("review_model", epoch))
    # 训练思路，一批中文一批英文，Q判断语言，损失同GAN, P只判断英文数据，损失同情感分类器


def evaluate(epoch=9):
    result = []
    with open("pkl/en_reviews_embedding.pkl", "rb") as f:
        en_embedding = pickle.load(f)
    with open("pkl/en_reviews.pkl", "rb") as f:
        en_set = pickle.load(f)
    en_data = torch.tensor(en_set["data"]).long()
    en_target = torch.tensor(en_set["target"]).long()
    en_length = torch.tensor(en_set["len"]).long()
    en_lang = en_set["lang"]
    with open("pkl/cn_reviews_embedding.pkl", "rb") as f:
        ch_embedding = pickle.load(f)
    with open("pkl/cn_reviews.pkl", "rb") as f:
        ch_set = pickle.load(f)
    ch_data = torch.tensor(ch_set["data"]).long()
    ch_target = torch.tensor(ch_set["target"]).long()
    ch_lang = ch_set["lang"]
    ch_length = torch.tensor(ch_set["len"]).long()



    en_embedding = embedding(en_lang.n_words, en_embedding, EMB_DIM).to(device)
    ch_embedding = embedding(ch_lang.n_words, ch_embedding, EMB_DIM).to(device)
    F = Feature_extractor(EMB_DIM, HID_DIM, 2, 0.2, BID)

    if BID:
        P = SentimentClassifier(2, HID_DIM * 2, 2, 0.2)
        Q = LanguageDetector(2, HID_DIM * 2, 0.2)
    else:
        P = SentimentClassifier(2, HID_DIM, 1, 0.2)
        Q = LanguageDetector(2, HID_DIM, 0.2)

    F, P, Q = F.to(device), P.to(device), Q.to(device)


    F_name="review_model/netF_epoch_%d.pth"%(epoch)
    P_name="review_model/netP_epoch_%d.pth"%(epoch)
    Q_name="review_model/netQ_epoch_%d.pth"%(epoch)
    F.load_state_dict(torch.load(F_name))
    P.load_state_dict(torch.load(P_name))
    Q.load_state_dict(torch.load(Q_name))
    batch_size = 1000
    en_batch, ch_batch, en_label,en_batch_len,ch_batch_len = get_batch_data_fast(batch_size, en_data, ch_data, en_target,en_length,ch_length)  # 分别取1000条双语数据
    en_batch, ch_batch, en_label,en_batch_len,ch_batch_len = en_batch.to(device),\
                                                             ch_batch.to(device), en_label.to(device), en_batch_len.to(device),ch_batch_len.to(device)
    TP, FP, TN, FN = 0, 0, 0, 0
    with torch.no_grad():
        en_batch = en_embedding(en_batch)
        en_F = F(en_batch,en_batch_len)
        en_predict = P(en_F)
        en_pre_label = en_predict.argmax(axis=1)
        acc = (en_label == en_pre_label).sum() / batch_size
        print("Accuracy on source language", acc)
        for i in range(BATCH_SIZE):
            if en_pre_label[i].item() == 1 and en_label[i].item() == 1:
                TP += 1
            elif en_pre_label[i].item() == 1 and en_label[i].item() == 0:
                FP += 1
            elif en_pre_label[i].item() == 0 and en_label[i].item() == 0:
                TN += 1
            elif en_pre_label[i].item() == 0 and en_label[i].item() == 1:
                FN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        print("Precision:%f  Recall:%f  F1 score:%f" % (precision, recall,F1))
    ch_batch, en_batch, ch_label,ch_batch_len,en_batch_len = get_batch_data_fast(batch_size, ch_data, en_data, ch_target,ch_length,en_length)
    ch_batch, en_batch, ch_label,ch_batch_len,en_batch_len = \
        ch_batch.to(device), en_batch.to(device), ch_label.to(device),ch_batch_len.to(device),en_batch_len.to(device)
    result.append(acc.item())
    with torch.no_grad():
        ch_batch = ch_embedding(ch_batch)
        ch_F = F(ch_batch,ch_batch_len)
        ch_predict = P(ch_F)
        ch_pre_label = ch_predict.argmax(axis=1)
        acc = (ch_label == ch_pre_label).sum() / batch_size
        print("Accuracy on target language", acc)
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(BATCH_SIZE):
            if ch_pre_label[i].item() == 1 and ch_label[i].item() == 1:
                TP += 1
            elif ch_pre_label[i].item() == 1 and ch_label[i].item() == 0:
                FP += 1
            elif ch_pre_label[i].item() == 0 and ch_label[i].item() == 0:
                TN += 1
            elif ch_pre_label[i].item() == 0 and ch_label[i].item() == 1:
                FN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        result.append(acc.item())
        print("Precision:%f  Recall:%f  F1 score:%f" % (precision, recall,F1))
        return result
if __name__ == '__main__':
    '''data=torch.zeros(32,10).long()
    em = torch.nn.Embedding(50,300)
    F = Feature_extractor(300,250)
    S = SentimentClassifier(2,500,2,0.2)
    L = LanguageDetector(2,500,0.2)
    data=em(data)
    print(data.size())
    data=F(data)
    print(data.size())
    data1=S(data)
    print(data1.size())
    data2 = L(data)
    print(data2.size())'''
    #train()
    acc_1=[0]
    acc_2=[0]

    batch=[i for i in range(11)]
    for i in range(10):
        result=evaluate(i)
        acc_1.append(result[0])
        acc_2.append((result[1]))
    plt.figure()
    plt.plot( batch,acc_1, c='red', marker='o')
    plt.plot( batch, acc_2, c='blue', marker='v')
    plt.legend(["English dataset", "Chinese dataset"])
    plt.xlabel("Epoch number")
    plt.ylabel("Evaluate Accuracy")

    plt.savefig(r'result/{}.png'.format("review_transfer"))
    plt.show()
