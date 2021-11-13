from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# import torch
import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.test.utils import datapath, get_tmpfile, common_texts
from gensim.models import KeyedVectors, Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
import wikipedia as w
from wikipedia.exceptions import WikipediaException
from articles import arr
import csv


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
    atricles_num = len(arr)
    data = open('data.csv', 'a', newline='')
    writer = csv.writer(data)
    writer.writerow(('sentence', 'isdefault'))
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
    # gen_model = Word2Vec(sentences=common_texts, vector_size=100, window=3, min_count=1, workers=4)
    # gen_model = Word2Vec.load('gen_model.txt')
    # gen_model.train([['known', 'well']], total_examples=1, epochs=1)
    # gen_model.save('gen_model.model')

    sim = model.most_similar('axiom')
    for s in sim:
        print(str(s))
    result = model.most_similar(positive=['woman', 'king'], negative=['man'])
    print("{}: {:.4f}".format(*result[0]))
    print(analogy('japan', 'japanese', 'russia'))

    visual_pca(model, words=['sex', 'wife', 'wine', 'brandy', 'spaghetti', 'hamburger', 'pizza', 'frog',
                             'ape', 'germany', 'france', 'israel', 'italy', 'school', 'homework', 'college'], sample=50)
    cleaned = []
    for i in range(atricles_num):
        art = get_wiki_content(i).lower().strip()
        for line in art.split('\n'):
            if line == "== references ==":
                break

            if line and not "==" in line:
                cleaned.append(line)
                print(line)


    with open('model.pkl', 'wb') as f:
        pkl.dump(model, f, protocol=pkl.HIGHEST_PROTOCOL)
