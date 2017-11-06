import sys
import csv
import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

def preprocess(dialogs):
    doc_set = []
    for folder in os.listdir(dialogs):
        folder_path = os.path.join(dialogs, folder)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            with open(file_path) as tsvfile:
                tsvreader = csv.reader(tsvfile, delimiter='\t')
                for line in tsvreader:
                    doc_set.append(line[-1])
    return doc_set


def train(doc_set, num_topics=10):
    tokenizer = RegexpTokenizer(r'\w+')

    # create English stop words list
    stop_words = set(stopwords.words('english'))

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # Create wordNetLemmatizer only extract NN and NNS to improve accuracy
    lem = WordNetLemmatizer()

    # list for tokenized documents in loop
    stemmed_docs = []

    # loop through document list
    for i in range(len(doc_set)):
        # clean and tokenize document string
        lower_case = doc_set[i].lower()
        tokens = tokenizer.tokenize(lower_case)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in stop_words]

        # extract only noun
        tagged_text = nltk.pos_tag(stopped_tokens)

        words = []

        for word, tag in tagged_text:
            words.append({"word": word, "pos": tag})

        nouns_tokens = [ lem.lemmatize(word["word"]) for word in words if word["pos"] in ["NN", "NNS"]]

        # stem tokens
        stemmed_tokens = [p_stemmer.stem(token) for token in nouns_tokens]

        # add tokens to list
        stemmed_docs.append(stemmed_tokens)
        if i%100000 == 0:
            print('[' + str(i)+ '/' + str(len(doc_set)) + ']')

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(stemmed_docs)
    print('finish creating dictionary')

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(doc) for doc in stemmed_docs]
    # generate LDA model
    print('start training ldamodel')
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=2)
    print('finish training ldamodel')
    return ldamodel, dictionary


def predict(ldamodel, dictionary, query):
    tokenizer = RegexpTokenizer(r'\w+')

    # create English stop words list
    stop_words = set(stopwords.words('english'))

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # Create wordNetLemmatizer only extract NN and NNS to improve accuracy
    lem = WordNetLemmatizer()

    # clean and tokenize document string
    lower_case = query.lower()
    tokens = tokenizer.tokenize(lower_case)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in stop_words]

    # extract only noun
    tagged_text = nltk.pos_tag(stopped_tokens)

    words = []

    for word, tag in tagged_text:
        words.append({"word": word, "pos": tag})

    nouns_tokens = [lem.lemmatize(word["word"]) for word in words if word["pos"] in ["NN", "NNS"]]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(token) for token in nouns_tokens]

    query_bow = dictionary.doc2bow(stemmed_tokens)

    print('This new query belongs to topic No.' + str(ldamodel.get_document_topics(query_bow)[0][0]) + ', with probability ' + str(ldamodel.get_document_topics(query_bow)[0][1]))


if __name__ == '__main__':
    flag = sys.argv[1]
    num_topics = 10
    num_words = 4
    if flag == 'train':
        dialogs = sys.argv[2]
        doc_set = preprocess(dialogs)
        print('Finish preprocessing')
        ldamodel, dictionary = train(doc_set, num_topics)
        if not os.path.exists('models'):
            os.mkdir('models')
        ldamodel.save('models/lda.model')
        dictionary.save('models/corpus.dict')
        print('Finish saving ldamodel and dictionary')
        print(ldamodel.print_topics(num_topics=num_topics, num_words=num_words))
    elif flag == 'test':
        if not os.path.exists('models/lda.model'):
            print('You need to run train mode first or put ldamodel in the same directory')
        else:
            ldamodel = models.LdaModel.load('models/lda.model')
            dictionary = corpora.Dictionary.load('models/corpus.dict')
            query = sys.argv[2]
            predict(ldamodel, dictionary, query)
    else:
        print('flag can either be train or test')
