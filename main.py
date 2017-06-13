import pandas as pd
import nltk
import gensim

# Prepare data for training
df = pd.read_csv('corpus/random/jokes.csv');
x = df['Question'].values.tolist()
y = df['Answer'].values.tolist()
corpus = x + y

# Tokenize data for word2vec creation
tok_corp = [nltk.word_tokenize(sent) for sent in corpus]

# Creating vectors from the words
model = gensim.models.Word2Vec(tok_corp, min_count=1, size=32)

# Saving and retrieving the trained model (for sanity checks)
# model.save('saved/model-jokes')
trained_model = gensim.models.Word2Vec.load('saved/model-jokes')

# Testing the model
print(trained_model.wv.most_similar('boy'))
