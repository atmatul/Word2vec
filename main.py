import os
import nltk
import gensim
import helper.clean as cleaner
from tqdm import tqdm

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

# Creating vectors from the words
model = gensim.models.Word2Vec(tokenized_corpus, min_count=3, size=32)

# Saving and retrieving the trained model (for sanity checks)
model.save('saved/model-shiva')

# -- Comment before here to use pretrained-data
trained_model = gensim.models.Word2Vec.load('saved/model-shiva')

# Testing the model
print(trained_model.wv.most_similar('Shiva'))
