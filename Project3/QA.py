import numpy as np
import nltk
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import sys  
import codecs

reload(sys)  
sys.setdefaultencoding('utf8')


class QASystem():
	def __init__(self):
		self.questions = []
		self.documents = {}
		self.corpus = {}


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
		print("Parsing Documents")
		for question in self.questions:
			document_location = "/documents/" + str(question[0])
			documents = ['' for i in xrange(0,100)]
			for doc in os.listdir(os.getcwd()+document_location):
				doc_text = open(os.getcwd()+document_location+'/'+doc)
				text = [line.replace('\r\n','')  for line in doc_text if not ('<P>' in line or '</P>' in line)]
				extracted_text = self.extract_text(text)
				text = ' '.join(extracted_text)	
				documents[int(doc)-1] = ''.join([i if ord(i) < 128 else ' ' for i in text])			
			self.corpus[question[0]] = documents
		
		print("Done parsing documents")
		


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
		return text[text_begin+1:text_end]


	# method that takes in list of documents, sentence tokenizes and concatenates the sentences to one list
	# also returns the number of sentences in each document
	def doc_sent_tokenizer(self, corpus):
		all_sentences = []
		doc_sentence_lengths = []
		for doc in corpus:
			doc_toke = sent_tokenize(doc)
			all_sentences = all_sentences + doc_toke
			doc_sentence_lengths.append(len(doc_toke))

		return all_sentences, doc_sentence_lengths

	# method that takes sent_indexes and length of document list by sentence and returns index
	# need to take indexes from passage retrieval and put in here
	def sent_index_to_doc_index(self, sent_idexes, len_documents_list):
		pass



	# method that converts sentence array indices into document indicies
	def compute_document_id(self,sentence_idxs,document_lengths):
		doc_rel_indices = []
		prev = 0
		sentence_pos = []
		for text_length in document_lengths:
			doc_rel_indices.append(prev+text_length)
			prev = doc_rel_indices[-1]
		for idx in sentence_idxs:
			if idx <= doc_rel_indices[0]:
				sentence_pos.append(0)
			elif doc_rel_indices[-2]<=idx:
				sentence_pos.append(len(doc_rel_indices))

			else:
				pos = 0
				while(idx<=doc_rel_indices[pos]):
					pos = pos+1
				sentence_pos.append(pos-1)
		return sentence_pos

	# method that computes the best answers using only passage retrival for each question
	def compute_answers(self):
		question_ids = []
		doc_ids = {}
		answers_ids = {}
		for question in self.questions:
			print("Answering Question: " + str(question[0]))
			question_ids.append(question[0])
			all_sentences,doc_sentence_lengths = self.doc_sent_tokenizer(self.corpus[question[0]])
			top_ten_words,indexes  = self.passage_retrieval(question[1],all_sentences,10)

			doc_ids[question[0]] = self.compute_document_id(indexes,doc_sentence_lengths)
			answers_ids[question[0]] = top_ten_words
		question_ids.sort()
		self.create_answers(question_ids,doc_ids,answers_ids)

	# method to compute cosine similiarity of the vector space between the query and documents
	# assuming query is the first item in the list of documents
	def passage_retrieval(self, query, corpus, n_items):
		corpus.insert(0, query)
		corpus = np.array(corpus)
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
	# param: doc_ids = dictionary with list values of document ids for each question, key is the question_id
	# param: answers = dictionary with list values of answers similiar to doc_ids list
	def create_answers(self, question_ids, doc_ids, answers):
		string = ""
		for q in question_ids:
			for x in range(0, 5):
				string += '{0} {1} {2}\n'.format(str(q), str(doc_ids[q][x]), str(answers[q][x]))
		with open("answer.txt", "w") as text_file:
			text_file.write(string)






def main():
	qa_system = QASystem()
	qa_system.parse_questions('question.txt')
	qa_system.parse_documents()
	qa_system.compute_answers()



if __name__ == '__main__':
	main()
