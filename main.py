import os
import nltk
import gensim
import pandas as pd
import helper.clean as cleaner
from tqdm import tqdm
import sklearn.manifold as skmanifold
import seaborn as sns

# Prepare data for training
_relative_path = 'corpus/shiva/'
corpus = []
for filename in os.listdir(os.path.join(os.getcwd(), _relative_path)):
    _filepath = os.path.join(_relative_path, filename)
    # df = pd.read_csv(_filepath, sep=None)
    # corpus = corpus + df['Raw_text']
    with open(_filepath) as file:
        corpus = corpus + file.read().split('\n')

# Tokenize and clean data for word2vec creation
print('\nTokenizing words:')
tokenized_corpus = [nltk.word_tokenize(sent) for sent in tqdm(corpus)]
tokenized_corpus_no_punc = cleaner.clean_punctuations(tokenized_corpus)
tokenized_corpus_no_stop = cleaner.clean_stopwords(tokenized_corpus_no_punc)

# Creating vectors from the wordss
model = gensim.models.Word2Vec(tokenized_corpus, min_count=3, size=32)

# Saving and retrieving the trained model (for sanity checks)
model.save('saved/model-shiva-test.w2v')

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Comment before here to use pretrained-data

trained_model = gensim.models.Word2Vec.load('saved/model-shiva.w2v')

# Testing the model
# print("Checking the association for 'Shiva':\n", trained_model.wv.most_similar('Shiva'))

# Plotting the trained word-vectors
print("Training model...")
tsne = skmanifold.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = trained_model.wv.syn0
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

# Generating the co-ordinates for plotting
print("Generating co-ordinates...")
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[trained_model.wv.vocab[word].index])
            for word in trained_model.wv.vocab
        ]
    ], columns=["word", "x", "y"])
_chartpath = os.path.join('data', 'chart-shiva.csv')
points.to_csv(_chartpath)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Comment before here to plot from saved-data

chart_points = pd.read_csv(_chartpath)

# Checking the most-frequent words
# print("Most-frequent words:\n", points.head(10))

# Plotting the actual data, finally!
sns.set_context("poster")
# sns.lmplot("x", "y", data=points, figsize=(20, 12))
chart_points.plot.scatter("x", "y", s=10, figsize=(20, 12))
for i, txt in enumerate(chart_points.word):
    sns.plt.annotate(txt, (chart_points.x[i], chart_points.y[i]), fontsize=10)
sns.plt.show()
