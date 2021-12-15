from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import wikipedia as w
import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.test.utils import datapath, get_tmpfile, common_texts
from gensim.models import KeyedVectors, Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from wikipedia.exceptions import WikipediaException
from articles import arr, resoning_words
import csv
import re


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
    articles_num = 4
    data = open('data.csv', 'a', newline='')
    writer = csv.writer(data)
    # writer.writerow(('sentence', 'isdefault'))
    data.close()
    model = ''
    if not os.path.isfile('model.pkl'):
        glove_file = datapath(f'{os.getcwd()}/glove.6B.100d.txt')
        word2vec_glove_file = get_tmpfile('glove.6B.100d.word2vec.txt')
        glove2word2vec(glove_file, word2vec_glove_file)
        model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
    else:
        with open('model.pkl', 'rb') as f:
            model = pkl.load(f)

    sim = model.most_similar('axiom')
    for s in sim:
        print(str(s))
    result = model.most_similar(positive=['woman', 'king'], negative=['man'])
    print("{}: {:.4f}".format(*result[0]))
    print(analogy('japan', 'japanese', 'russia'))

    visual_pca(model, words=['sex', 'wife', 'wine', 'brandy', 'spaghetti', 'hamburger', 'pizza', 'frog',
                             'ape', 'germany', 'france', 'israel', 'italy', 'school', 'homework', 'college'], sample=50)
    cleaned = {}
    MIN = 60
    for i in range(articles_num):
        art = get_wiki_content(i).lower().strip()
        cleaned[i] = []
        for j, line in enumerate(art.split('\n')):
            if line == "== references ==":
                break

            sizer = len(line)
            if line and not "==" in line and sizer > MIN:
                sentences = re.split(r'[.?!]\s* ', line)
                if sentences[-1]:
                    cleaned[i].append(sentences)
                else:
                    cleaned[i].append(sentences[:-1])

                # cleaned.extend([sentence for sentence in line.split(])
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

    for pred in preds:
        inds = pred[-1]
        par_len = len(cleaned_sorted[inds[0]])
        sent_len = len(cleaned_sorted[inds[0]][inds[1]])
        print(f'Article: {inds[0]}/{articles_num}, Paragraph: {inds[1]}/{par_len}, Sentence: {inds[2]}/{sent_len}:\n{pred[0]}\n')

    with open('model.pkl', 'wb') as f:
        pkl.dump(model, f, protocol=pkl.HIGHEST_PROTOCOL)
