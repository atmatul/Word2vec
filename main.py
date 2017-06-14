import os
import nltk
import gensim
import pandas as pd
import helper.clean as cleaner
from tqdm import tqdm
import sklearn.manifold as skmanifold
import seaborn as sns


def create_model(relative_path, model_path):
    corpus = []
    for filename in os.listdir(os.path.join(os.getcwd(), relative_path)):
        _filepath = os.path.join(relative_path, filename)
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
    model = gensim.models.Word2Vec(tokenized_corpus_no_stop, min_count=3, size=32)
    return model


def train_model(trained_model):

    # Testing the model
    # print("Checking the association for 'Shiva':\n", trained_model.wv.most_similar('Shiva'))

    # Plotting the trained word-vectors
    print("\nTraining model...")
    tsne = skmanifold.TSNE(n_components=2, random_state=0)
    all_word_vectors_matrix = trained_model.wv.syn0
    all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

    # Generating the co-ordinates for plotting
    print("\nGenerating co-ordinates...")
    points = pd.DataFrame(
        [
            (word, coords[0], coords[1])
            for word, coords in [
                (word, all_word_vectors_matrix_2d[trained_model.wv.vocab[word].index])
                for word in trained_model.wv.vocab
            ]
        ], columns=["word", "x", "y"])
    return points


def visualize_charts(chart_points, annotations=False):
    # Checking the most-frequent words
    # print("Most-frequent words:\n", points.head(10))

    # Plotting the actual data, finally!
    sns.set_context("poster")
    # sns.lmplot("x", "y", data=points, figsize=(20, 12))
    chart_points.plot.scatter("x", "y", s=10, figsize=(16, 9))
    if annotations:
        for i, txt in enumerate(chart_points.word):
            sns.plt.annotate(txt, (chart_points.x[i], chart_points.y[i]), fontsize=10)
    sns.plt.show()


if __name__ == '__main__':
    # Prepare data for training
    _relative_path = os.path.join('corpus', 'shiva')
    _model_path = os.path.join('saved', 'model-shiva.w2v')
    _chart_path = os.path.join('data', 'chart-shiva.csv')

    # Creating the model
    _model = create_model(_relative_path, _model_path)
    # Saving the trained model
    _model.save(_model_path)
    # Training the model
    _trained_model = _model   # or
    # _trained_model = gensim.models.Word2Vec.load('saved/model-shiva.w2v')
    _points = train_model(_trained_model)
    # Saving the co-ordinates for visualization
    _points.to_csv(_chart_path)
    # Visualizing the model
    _chart_points = _points    # or
    # _chart_points = pd.read_csv(_chart_path)
    visualize_charts(_chart_points, annotations=True)
