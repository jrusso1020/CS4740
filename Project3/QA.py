import numpy as np
import nltk
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer


class QASystem():
	def __init__(self):
		self.questions = []
		self.documents = {}

	# Parse questions to be answered, store questions in self.questions as
	# a tuple of (question_idx,question)
	def parse_questions(self,filename):
		question_file = open(filename,'r')
		list_questions = [line.replace('\r\n','')  for line in question_file]
		i = 0
		questionIDX = 0
		while(i<len(list_questions)):
			if 'Number' in list_questions[i]:
				questionIDX = int(''.join(c for c in list_questions[i] if c.isdigit()))
			if 'Description' in list_questions[i]:
				question = list_questions[i+1]
				self.questions.append((questionIDX,question))
			i = i+1

	# Parse retrieved documents and extract text
	def parse_documents(self):
		for question in self.questions:
			document_location = "/documents/" + str(question[0])
			for doc in os.listdir(os.getcwd()+document_location):
				doc_text = open(os.getcwd()+document_location+'/'+doc)
				extracted_text = self.extract_text(doc_text)



	# helper method for extracting text segment
	def extract_text(self,text):
		text_begin = 0
		text_end = 0
		for i in xrange (0,len(text)):
			line = text[i]
			if '<TEXT>' in line:
				text_begin = i
			if '</TEXT>' in line:
				text_end = i


	# method to compute cosine similiarity of the vector space between the query and documents
	# assuming query is the first item in the list of documents
	def passage_retrieval(self, query, corpus, n_items):
		corpus.insert(0, query)
		vect = TfidfVectorizer(min_df=1)
		tfidf = vect.fit_transform(corpus)

		similiarity_matrix = (tfidf * tfidf.T).A

		#get just the documents not including query
		query_doc_similiarities = similiarity_matrix[0, 1:]


		indexes = query_doc_similiarities.argsort()[-n_items:][::-1]
		top_n = corpus[indexes + 1]
		top_ten_words = []
		for sent in top_n:
			split = sent.split()
			top_ten_words.append(split[:10])

		return top_ten_words, indexes

	# need method to convert list of sentence indexes to document ID's for a question
	def indexes_to_doc_id(self, indexes):
		pass






def main():
	qa_system = QASystem()
	qa_system.parse_questions('question.txt')
	qa_system.parse_documents()


if __name__ == '__main__':
	main()
