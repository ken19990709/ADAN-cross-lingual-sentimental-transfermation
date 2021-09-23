import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from word_embedding import *
import spacy
import torch
import pickle
import re
import jieba
import copy

max_length = 20
sns.set_style('darkgrid')


def word2i(word, w2i):
    if word in w2i:
        return w2i[word]
    else:
        return 3  # 找不到即为UNK


def build_weibo_dataset(use_words=True):
    data = pd.read_csv('../dataset/weibo/weibo_senti_100k.csv',
                       encoding='UTF-8')

    ch_set = data["review"].tolist()
    labels = data["label"].tolist()

    nlp = spacy.load("zh_core_web_lg")  # 导入中文分析库

    for i in range(len(ch_set)):
        temp = ch_set[i]

        try:
            temp = re.sub(r'(@\w*)', '', temp)
            temp = re.sub(r"【.*?】", "", temp)
            temp = re.sub(r"『.*?』", "", temp)
            temp = re.sub(r"[.*?]", "", temp)
            temp = re.sub(r"[{].*?[}]", "", temp)
            temp = re.sub(r"[[].*?[]]", "", temp)
            temp = temp.strip()

            temp = re.sub("[^\u4e00-\u9fa5]", "", temp)

        except:
            print(i)

        if use_words:
            temp = nlp(temp)
            temp = [word.text for word in temp]

        else:
            temp = [word for word in temp]

        if temp == None or len(temp) < 5:
            ch_set[i] = None
            continue
        ch_set[i] = temp
    ch_set_cp = []
    labels_cp = []
    while (1):
        try:
            ch_set.remove(None)

        except:
            break
    sen_length = [len(x) for x in ch_set]
    sns.boxplot(sen_length)
    plt.show()

    for i in range(len(ch_set)):
        temp = ch_set[i]
        if len(temp) > (max_length * 1.5) or len(temp) < (max_length * 0.5):
            pass
        elif (len(temp) >= max_length):
            temp = temp[:max_length]
            label = labels[i]
            ch_set_cp.append(temp)
            labels_cp.append(label)

        else:
            temp.append("EOS")  # 添加EOS
            for fill in range(max_length - len(temp)):
                temp.append(" ")
            label = labels[i]
            ch_set_cp.append(temp)
            labels_cp.append(label)

    '''sns.boxplot(sen_length)
    plt.show()'''
    cn_lang = Lang("cn")
    # print(data)
    for cn in ch_set_cp:
        cn_lang.addSentence(cn)
    cn_lang.drop_single_word()
    w2i = cn_lang.word2index
    data = [[word2i(word, w2i) for word in words] for words in ch_set_cp]

    result = {"data": data, "target": labels_cp, "lang": cn_lang}
    with open("pkl/weibo_100000_test.pkl", "wb") as f:
        pickle.dump(result, f)


# build_weibo_dataset()


def format_text(df, col):
    # Remove @ tags
    comp_df = df.copy()
    comp_df[col] = comp_df[col].str.lower()
    # remove all the punctuation
    comp_df[col] = comp_df[col].str.replace(r'(@\w*)', '')

    # Remove URL
    comp_df[col] = comp_df[col].str.replace(r"http\S+", "")
    comp_df[col] = comp_df[col].str.replace(r"https\S+", "")
    comp_df[col] = comp_df[col].str.replace(r"www.\S+", "")
    # Remove # tag and the following words
    comp_df[col] = comp_df[col].str.replace(r'#\w+', "")
    comp_df[col] = comp_df[col].str.replace(r'&quot', "")
    comp_df[col] = comp_df[col].str.replace(r'&gt', ">")
    comp_df[col] = comp_df[col].str.replace(r'&lt', "<")
    comp_df[col] = comp_df[col].str.replace(r'&amp', "and")
    # Remove useless word
    # comp_df[col] = comp_df[col].str.replace(r'contd', "")
    comp_df[col] = comp_df[col].str.replace(r'missn', "miss")
    comp_df[col] = comp_df[col].str.replace(r'sorryi', "sorry")
    comp_df[col] = comp_df[col].str.replace(r'homee', "home")
    comp_df[col] = comp_df[col].str.replace(r'everyonee', "everyone")
    comp_df[col] = comp_df[col].str.replace(r'nightt', "night")
    comp_df[col] = comp_df[col].str.replace(r'todayyy', "today")
    comp_df[col] = comp_df[col].str.replace(r'todayi', "today")
    comp_df[col] = comp_df[col].str.replace(r'loveyou', "love you")

    comp_df[col] = comp_df[col].str.replace(r'todayis', "today is")
    comp_df[col] = comp_df[col].str.replace(r'blogtv', "blog tv")
    comp_df[col] = comp_df[col].str.replace(r'howre', "how are")
    comp_df[col] = comp_df[col].str.replace(r'youre', "you are")

    # Remove all non-character
    comp_df[col] = comp_df[col].str.replace(r"[^a-zA-Z ]", "")

    # Remove extra space
    comp_df[col] = comp_df[col].str.replace(r'( +)', " ")
    comp_df[col] = comp_df[col].str.strip()

    # Change to lowercase

    return comp_df


def get_word_list(dataframe, clear_stopwords=False):
    df = dataframe.copy()

    # lemmatize the word
    nlp = spacy.load("en")

    df.text = df.text.apply(lambda x: x.split(" "))
    # 暂时不管停用
    if clear_stopwords:
        df.text = df.text.apply(lambda x: [i for i in x if not i.is_stop == True])

    # combined into a sentence again
    # df.text = df.text.apply(lambda x: [i.text for i in x])

    # df.text = df.text.apply(lambda x: ' '.join(x))

    word_list = df.text.tolist()

    return word_list


def build_twitter_data_set():
    comp_df = pd.read_csv('../dataset/twitter/training.1600000.processed.noemoticon.csv',
                          # comp_df=pd.read_csv('../dataset/twitter/training.20000.csv',
                          encoding='ISO-8859-1',
                          header=None)
    comp_df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    comp_df.target = comp_df.target.map({4: 1, 0: 0})
    # sns.countplot(comp_df.target)
    # plt.show()
    formated_comp_df = format_text(comp_df, 'text')
    formated_comp_df.drop(['id', 'date', 'flag', 'user'], axis=1, inplace=True)
    # print(formated_comp_df.head())

    target = formated_comp_df["target"].tolist()

    sen_length = [len(x) for x in formated_comp_df.text.tolist()]
    # sns.boxplot(sen_length)
    # plt.title('Distribution of sentence length')
    # plt.show()

    data = get_word_list(formated_comp_df, clear_stopwords=False)
    # print(data[0:10])
    en_set_cp = []
    labels_cp = []
    for i in range(len(data)):
        temp = data[i]
        if len(temp) > (max_length * 1.5) or len(temp) < (max_length * 0.5):
            pass
        elif (len(temp) >= max_length):
            temp = temp[:max_length]
            label = target[i]
            en_set_cp.append(temp)
            labels_cp.append(label)

        else:
            temp.append("EOS")
            for fill in range(max_length - len(temp)):
                temp.append(" ")
            label = target[i]
            en_set_cp.append(temp)
            labels_cp.append(label)

    en_lang = Lang("en")
    # print(en_set_cp[0:10])
    for en in en_set_cp:
        en_lang.addSentence(en)
    en_lang.drop_single_word()
    w2i = en_lang.word2index
    data = [[word2i(word, w2i) for word in words] for words in en_set_cp]

    # print(en_lang.n_words)
    result = {"data": data, "target": labels_cp, "lang": en_lang}
    with open("pkl/twitter_1600000_test.pkl", "wb") as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    build_twitter_data_set()
    build_weibo_dataset()

# print(result["lang"].word2index)
