from sklearn.decomposition import PCA
import wikipedia as w
import pickle as pkl
import os
import numpy as np
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from wikipedia.exceptions import WikipediaException
from articles import arr, resoning_words
from nltk.tokenize import sent_tokenize
from mongo import coll


def get_wiki_content(index) -> list:
    # conts = []
    a = False
    try:
        cont = w.page(str(arr[index])).content
        return cont
        # print(cont)
        # conts.append(cont)
    except WikipediaException:
        a = True


def analogy(x1, x2, y1):
    res = model.most_similar(positive=[y1.lower(), x2.lower()], negative=[x1.lower()])
    return res[0][0]


def visual_pca(model, words=None, sample=0):
    if not words:
        if sample > 0:
            words = np.random.choice(list(model.index_to_key), sample)
        else:
            words = [word for word in model.vocab]
    word_vecs = np.array([model[w] for w in words])
    two_dim = PCA().fit_transform(word_vecs)[:, :2]
    plt.figure(figsize=(6, 6))
    plt.scatter(two_dim[:, 0], two_dim[:, 1], edgecolors='k', c='r')
    for word, (x, y) in zip(words, two_dim):
        plt.text(x + 0.05, y + 0.05, word)
    plt.show()


if __name__ == "__main__":
    articles_num = 20
    model = ''
    col_name = 'text'
    df = pd.DataFrame(columns=[col_name])

    cleaned = {}
    MIN = 60
    for i in range(articles_num):
        art = str(get_wiki_content(i)).lower().strip()
        cleaned[i] = []
        for j, line in enumerate(art.split('\n')):
            if line == "== references ==":
                break

            sizer = len(line)
            if line and not "==" in line and sizer > MIN:
                sentences = sent_tokenize(line)
                sentences = list(sentences)
                for sen in sentences:
                    tmp = ''
                    if not re.match('^[A-Za-z0-9.,:;!?()]+$', sen):  # Skip not english sentences
                        continue
                    sen = ''.join([x for x in sen if x.isalnum() or x in string.punctuation or x == ' '])
                    size = len(sen)
                    for k in range(size):
                        if sen[k] == '.':
                            if k + 2 < size and (not str(sen[k + 1]).isalpha() and not str(sen[k + 1]).isdigit()) and \
                                    sen[k + 2] != '.':
                                tmp = sen[k + 1:]
                                sentences.append(tmp)
                                sen = sen[:k + 1]
                                break
                if sentences[-1]:
                    cleaned[i].append(sentences)
                else:
                    cleaned[i].append(sentences[:-1])

                print(f'iteration {j} (size = {sizer}): {line}\n')

    cleaned_sorted = {}
    preds = []
    for key in list(cleaned.keys()):
        parags = cleaned[key]
        cleaned_sorted[key] = []
        for j in range(len(parags)):
            cleaned_sorted[key].append([])
            parag = parags[j]
            for z in range(len(parag)):
                cleaned_sorted[key][j].append((parag[z], (int(key), int(j), int(z))))
                if [x for x in resoning_words if x in parag[z]]:
                    preds.append((parag[z], (int(key), int(j), int(z))))

    for ind, pred in enumerate(preds):
        # coll.insert_one({"sentence": str(pred[0]), "ans": 1, "checked": False})
        df.loc[ind] = [pred[0]]  # For AWS MTurk
        # print(df)
        inds = pred[-1]
        par_len = len(cleaned_sorted[inds[0]])
        sent_len = len(cleaned_sorted[inds[0]][inds[1]])
        print(
            f'Article: {inds[0]}/{articles_num}, Paragraph: {inds[1]}/{par_len}, Sentence: {inds[2]}/{sent_len}:\n{pred[0]}\n')

    df.to_csv(f'{os.getcwd()}/data{articles_num}.csv', index=False)
