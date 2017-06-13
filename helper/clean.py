import re
import string
from nltk.corpus import stopwords
from tqdm import tqdm


def clean_punctuations(tokenized_corpus):
    """ Removes punctuations
        Helps in further processing by cleaning unnecessary punctuation letters.
    :param tokenized_corpus:
    :return:
    """
    print('\nCleaning punctuations:')
    tokenized_corpus_no_punc = []
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    for sentence in tqdm(tokenized_corpus):
        if len(sentence) == 0:
            continue
        new_sentence = []
        for token in sentence:
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                new_sentence.append(new_token)
        tokenized_corpus_no_punc.append(new_sentence)
    return tokenized_corpus_no_punc


def clean_stopwords(tokenized_corpus):
    """ Removes stopwords
        Common stop-words such as 'a', 'an' etc. add no substantial information
        into the corpus and can be eleminated.
    :param tokenized_corpus:
    :return:
    """
    print('\nCleaning stop-words:')
    tokenized_docs_no_stop = []
    for doc in tqdm(tokenized_corpus):
        new_term_vector = []
        for word in doc:
            if word not in stopwords.words('english'):
                new_term_vector.append(word)
        tokenized_docs_no_stop.append(new_term_vector)
    return tokenized_docs_no_stop
