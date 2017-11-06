# NLP-topics
Latent Dirichlet allocation (LDA) is used to solve this problem, a topic model that generates topics based on word frequency from a set of documents. Since the data sets are huge ,costing too much RAM and  time to come to a result. I choose one tenth of the data as my training samples to show my code can return a reasonable result. 

The code can run on python3.6, with nltk, gensim libraries. 
In order to successfully run the code, one might need to download additional package for nltk by:
	import nltk
	nltk.download(‘stopwords’)
	nltk.download(‘wordnet’)
	nltk.download(‘averaged_perceptron_tagger’)
  
The comments in actual code explain the detail of code structure. The basic ideas are as follows:
First, I extract all context from The Ubuntu Dialogue Corpus as my training document set. Then I start preprocessing these data, for example, replacing them in lower number, removing stop words, stemming words. Especially, after several trials, observing that topic tends to be noun. Therefore, I take advantage of the POS tagging and extract only the NN and NNS from the document. After that, these lemmatized data are zipped into a dictionary passing into LDA model, in which I set the number of words to descript each topic is 4 and we get the top ten most popular topics. Finally, I save the model and the data.

The test mode is process in the same way. I first load the pretrained lda model and dictionary from /models/. And get the new query through argument and then do the same preprocessing procedure as training documents. Finally, Using the pretrained dictionary to get the bag of word of the stemmed query and ldamodel.get_document_topics was called to return the most related topic for that specific query. 

If you want to test the code, you can follow the syntax below:
For training mode:
python3 linc_project_new5.py train <path to the dataset folders: dialog>
For test query:
python3 linc_project_new5.py test <your query>
  
The output would be top 10 topics list. The first token of each sublist is the topic id and the second part is the weighted words descripting the topic.

I have spent about 3 hours to get the whole project up and run from scratch and spent about 4 hours to train 3 different versions of the ldamodel. I spent most of my time to wait for the result and find a way to deal with the memory issue and how to speed up the whole read in and training process. The time of predict a new result is just about one second.
