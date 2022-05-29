#
# if not os.path.isfile('model.pkl'):
#     glove_file = datapath(f'{os.getcwd()}/glove.6B.100d.txt')
#     word2vec_glove_file = get_tmpfile('glove.6B.100d.word2vec.txt')
#     glove2word2vec(glove_file, word2vec_glove_file)
#     model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
# else:
#     with open('model.pkl', 'rb') as f:
#         model = pkl.load(f)

# sim = model.most_similar('always')
# for s in sim:
#     print(str(s))
# result = model.most_similar(positive=['woman', 'king'], negative=['man'])
# print("{}: {:.4f}".format(*result[0]))
# print(analogy('japan', 'japanese', 'russia'))
#
# visual_pca(model, words=['sex', 'wife', 'wine', 'brandy', 'spaghetti', 'hamburger', 'pizza', 'frog',
#                          'ape', 'germany', 'france', 'israel', 'italy', 'school', 'homework', 'college'], sample=50)
# sentences = re.split(r'[.?!]\s* ', line)

# cleaned.extend([sentence for sentence in line.split(])

# with open('model.pkl', 'wb') as f:
#     pkl.dump(model, f, protocol=pkl.HIGHEST_PROTOCOL)
