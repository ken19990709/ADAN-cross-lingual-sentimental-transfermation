# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import pickle
import torch
from code.old import models
from text_RNN import sentimental_Model
from code.old.models import AE
import numpy as np
from sklearn.model_selection import train_test_split

def load_embedding_and_dictionary():
    with open("pkl/twitter_1600000_test.pkl", "rb") as f:
        en_data = pickle.load(f)
    with open("pkl/weibo_100000_test.pkl", "rb") as f:
        cn_data = pickle.load(f)
    en_lang = en_data["lang"]
    cn_lang = cn_data["lang"]
    with open("pkl/twitter_embedding_test.pkl", "rb") as f:
        en_embedding = pickle.load(f)
    with open("pkl/weibo_embedding_test.pkl", "rb") as f:
        cn_embedding = pickle.load(f)

    print(en_embedding.size())
    print(cn_embedding.size())

    return en_lang, cn_lang, en_embedding, cn_embedding
def adversarial_train():
    model = models.BiAAE(emb_dim=300, hid_G_dim=300, hid_D_dim=2500, d_lr=0.1, g_lr=0.1)
    en_lang, cn_lang, en_embedding, cn_embedding = load_embedding_and_dictionary()
    model.train(en_lang, cn_lang, en_embedding, cn_embedding)

def convert_batch_data(Batch_data,batch_size,embedding,my_AE,src="en"):
    embedding_machine = torch.nn.Embedding(embedding.size(0),300,_weight=embedding)
    tar_AE = AE(emb_dim=300, hid_dim=300)
    tar_AE.load_state_dict(torch.load("model/X_AE.pkl"))
    if torch.cuda.is_available():
        embedding_machine = embedding_machine.cuda()
        tar_AE=tar_AE.cuda()
    with torch.no_grad():
        result = embedding_machine(Batch_data)
        result =my_AE.encode(result)
        result = tar_AE.decode(result)


    return result

def sentiment_train(src="en",keep=False):#use English data to train the model
    with open("pkl/twitter_1600000_test.pkl", "rb") as f:
        result = pickle.load(f)
    en_lang = result["lang"]

    en_set = result["data"]
    label = result["target"]
    en_set=np.array(en_set)
    label=np.array(label)
    train_X, test_X, train_y, test_y = train_test_split(en_set, label, test_size=1000/label.shape[0], random_state=5)

    if keep:
        model = torch.load("model/text_RNN_transfer.pkl")
    else:
        model =sentimental_Model(en_lang.n_words)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    my_AE = AE(emb_dim=300, hid_dim=300)

    if src == "en":
        with open("pkl/twitter_embedding_test.pkl", "rb") as f:
            embedding = pickle.load(f)
            my_AE.load_state_dict(torch.load("model/X_AE.pkl"))
    else:
        with open("pkl/weibo_embedding_test.pkl", "rb") as f:
            embedding = pickle.load(f)
            my_AE.load_state_dict(torch.load("model/Y_AE.pkl"))
    batch_size = 256
    lr = 0.02
    epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9)
    if torch.cuda.is_available():
        train_X = torch.from_numpy(train_X).long()
        test_X = torch.from_numpy(test_X).long().cuda()
        train_y = torch.from_numpy(train_y).long()
        test_y = torch.from_numpy(test_y).long().cuda()
        my_AE = my_AE.cuda()

    best_loss = 1000
    early_stop=0
    # cuda
    for epoch in range(epochs):

        print("epoch:", epoch)
        for index in range(0, len(train_X), batch_size):
            if index + batch_size < len(train_X):
                train_data = train_X[index:index + batch_size]
                train_label = train_y[index:index + batch_size]
            else:
                train_data = train_X[index:]
                train_label = train_y[index:]

            # cuda

            optimizer.zero_grad()

            if torch.cuda.is_available():
                train_data = train_data.cuda()
                train_label = train_label.cuda()
            train_data = convert_batch_data(train_data,batch_size,embedding,my_AE)
            model.train()
            predict = model(train_data)
            predict_label = predict.argmax(axis=1)
            acc = (predict_label == train_label).sum() / batch_size

            loss = criterion(predict, train_label)
            loss.backward()
            optimizer.step()

            if index / batch_size % 1000 == 0 and index / batch_size % 2000 != 0:
                print("batch:%d | loss:%f | train_acc:%f" % (index / batch_size, loss, acc))
            elif index / batch_size % 2000 == 0:
                model.eval()
                test_acc = 0
                test_loss = 0

                for test_i in range(0, test_X.size(0), batch_size):
                    if test_i + batch_size < test_X.size(0):
                        test_data = test_X[test_i:test_i + batch_size]
                        test_label = test_y[test_i:test_i + batch_size]
                    else:
                        test_data = test_X[test_i:]
                        test_label = test_y[test_i:]

                    test_data=convert_batch_data(test_data,batch_size,embedding,my_AE)
                    predict_test = model(test_data)

                    predict_test_label = predict_test.argmax(axis=1)
                    test_loss += criterion(predict_test, test_label)
                    test_acc += (predict_test_label == test_label).sum() / test_y.size(0)
                if best_loss<test_loss:
                    scheduler.step()
                    early_stop+=1
                else:
                    early_stop = 0
                    best_loss = test_loss
                if(early_stop>=10):
                    print("early stop")
                    torch.save(model, 'model/text_RNN_transfer.pkl')
                    sys.exit()


                print("batch:%d | loss:%f | train_acc:%f | test_loss:%f | test_acc:%f " % (
                    index / batch_size, loss, acc, test_loss, test_acc))

        if epoch % 1 == 0:
            torch.save(model, 'model/text_RNN_transfer.pkl')

    pass
def sentiment_eval():#use Chinese data to evaluate the model

    with open("pkl/weibo_100000_test.pkl", "rb") as f:
        result = pickle.load(f)
    cn_lang = result["lang"]
    batch_size = 128
    cn_set = result["data"]
    label = result["target"]
    cn_set=np.array(cn_set)
    label=np.array(label)
    src="cn"
    my_AE = AE(emb_dim=300, hid_dim=300)

    with open("pkl/weibo_embedding_test.pkl", "rb") as f:
        embedding = pickle.load(f)
        my_AE.load_state_dict(torch.load("model/Y_AE.pkl"))
    right_predict=0
    model = torch.load("model/text_RNN_transfer.pkl")
    if torch.cuda.is_available():
        cn_set=torch.from_numpy(cn_set).long()
        label=torch.from_numpy(label).long()
        model = model.cuda()
        my_AE = my_AE.cuda()
        embedding = embedding.cuda()

    for index in range(0, len(cn_set), batch_size):
        if index + batch_size < len(cn_set):
            batch_data = cn_set[index:index + batch_size]
            batch_label = label[index:index + batch_size]
        else:
            batch_data = cn_set[index:]
            batch_label = label[index:]
        if torch.cuda.is_available():
            batch_data=batch_data.cuda()
            batch_label=batch_label.cuda()
        batch_data = convert_batch_data(batch_data,batch_size,embedding,my_AE,src="cn")


        predict=model(batch_data)

        predict_label = predict.argmax(axis=1)
        right_predict += (predict_label == batch_label).sum()
    print("The accuracy in chinese set is %f"%(right_predict/len(cn_set)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    model = models.BiAAE(emb_dim=300, hid_G_dim=300, hid_D_dim=2500, d_lr=0.1, g_lr=0.1, keep=True)
    en_lang, cn_lang, en_embedding, cn_embedding=load_embedding_and_dictionary()
    model.train(en_lang, cn_lang, en_embedding, cn_embedding)

    #sentiment_train()
    #sentiment_eval()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
