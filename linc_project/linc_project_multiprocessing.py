import sys
import csv
import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import multiprocessing
from multiprocessing import Pool


def read_in_docs(dialogs):
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


def data_preprocess(doc, tokenizer, stop_words, p_stemmer, lem):
    
    lower_case = doc.lower()
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

    # add tokens to list
    stemmed_docs.append(stemmed_tokens)

    return stemmed_tokens


def train(stemmed_docs, num_topics=10):
    # turn our tokenized documents into a id <-> term dictionary
    processed_docs = [res.get() for res in stemmed_docs]
    dictionary = corpora.Dictionary(processed_docs)
    print('finish creating dictionary')

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    # generate LDA model
    print('start training ldamodel')
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=2)
    print('finish training ldamodel')
    return ldamodel, dictionary


def predict(ldamodel, dictionary, stemmed_tokens):
    query_bow = dictionary.doc2bow(stemmed_tokens)
    print(ldamodel.get_document_topics(query_bow))


if __name__ == '__main__':
    flag = sys.argv[1]
    num_topics = 10
    num_words = 4
    if flag == 'train':
        dialogs = sys.argv[2]
        doc_set = read_in_docs(dialogs)
        print('Finish reading')
        # work_queue = []
        # list for tokenized documents in loop
        stemmed_docs = []
        tokenizer = RegexpTokenizer(r'\w+')
    
        # create English stop words list
        stop_words = set(stopwords.words('english'))
        
        # Create p_stemmer of class PorterStemmer
        p_stemmer = PorterStemmer()
        
        # Create wordNetLemmatizer only extract NN and NNS to improve accuracy
        lem = WordNetLemmatizer()
        # clean and tokenize document string

        # for doc in doc_set:
        #     work_queue.append(([doc, stemmed_docs], None))
        # mutithread preprocessing
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        for doc in doc_set:
            stemmed_docs.append(pool.apply_async(data_preprocess, ([doc, tokenizer, stop_words, p_stemmer, lem],)))
        #requests = threadpool.makeRequests(data_preprocess, work_queue)

        # [pool.putRequest(req) for req in requests]
        pool.close()
        pool.join()
        print('mission complete!')

        ldamodel, dictionary = train(stemmed_docs, num_topics)
        print(ldamodel.print_topics(num_topics=num_topics, num_words=num_words))
        if not os.path.exists('models'):
            os.mkdir('models')
        ldamodel.save('models/lda.model')
        dictionary.save('models/corpus.dict')
        print('Finish saving ldamodel and dictionary')
    elif flag == 'test':
        if not os.path.exists('models/lda.model'):
            print('You need to run train mode first or put ldamodel in the same directory')
        else:
            print('reading ldamodel from pretrain model')
            ldamodel = models.LdaModel.load('models/lda.model')
            dictionary = corpora.Dictionary.load('models/corpus.dict')
            query = sys.argv[2]
            stemmed_tokens = data_preprocess(query, tokenizer, stop_words, p_stemmer, lem)
            predict(ldamodel, dictionary, stemmed_tokens)
    else:
        print('flag can either be train or test')
