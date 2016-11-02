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

		if len(top_ten_words)<5:
			top_ten_words = top_ten_words + (["nil"] * (5-len(top_ten_words)))
			indexes = indexes + ([1] * (5-len(top_ten_words)))

		return top_ten_words, indexes

		# method to create answer.txt and save it
		# param: question_ids = one list of all question ids
		# param: doc_ids = list of lists of document ids for each question, index corresponds to same index of the question id in its list
		# param: answers = list of lists of answers similiar to doc_ids list
		# could possibly change doc_ids and answers to dictionaries, would just need to change code slightly
		def create_answers(self, question_ids, doc_ids, answers):
			string = ""
			for idx, q in enumerate(question_ids):
				for x in range(0, 5):
					string += '{0} {1} {2}\n'.format(str(q), str(doc_ids[idx][x]), str(answers[idx][x]))
			with open("answer.txt", "w") as text_file:
				text_file.write(string)






def main():
	qa_system = QASystem()
	qa_system.parse_questions('question.txt')
	qa_system.parse_documents()


if __name__ == '__main__':
	main()
