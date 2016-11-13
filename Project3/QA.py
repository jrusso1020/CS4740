import numpy as np
import nltk
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import sys
import codecs
from nltk.tag.stanford import StanfordNERTagger

reload(sys)
sys.setdefaultencoding('utf8')


class QASystem():
	def __init__(self):
		self.nerTagger =  StanfordNERTagger('stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
					   'stanford-ner/stanford-ner.jar',
					   encoding='utf-8')
		self.dateTegger = StanfordNERTagger('stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz',
					   'stanford-ner/stanford-ner.jar',
					   encoding='utf-8')
		self.questions = []
		self.corpus = {}
		self.taggedQuestions ={}
		self.answers_ids = {}
		self.tagged_passages  = {}
		self.finalAnswers = {}
		self.finalIdxs = {}
		self.doc_lengths = {}

	# tag each question with the expected output type
	def tag_questions(self):
		for tup in self.questions:
			if 'Where' in tup[1]:
				self.taggedQuestions[tup[0]] = 'LOCATION'
			elif 'Who' in tup[1]:
				self.taggedQuestions[tup[0]] = 'PERSON'
			else:
				self.taggedQuestions[tup[0]] = 'DATE'


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


	# method that converts sentence array indices into document indicies
	def compute_document_id(self,sentence_idxs,document_lengths):
		sentence_idxs = list(sentence_idxs)
		doc_ids = [0] * 5
		sorted_sentence_idxes = sorted(sentence_idxs)
		sum_lengths = 0
		index = 0
		for idx, doc in enumerate(document_lengths):
			if sum_lengths < sorted_sentence_idxes[index] and sum_lengths+doc>=sorted_sentence_idxes[index]:
				doc_ids[sentence_idxs.index(sorted_sentence_idxes[index])] = idx
				index +=1
				if index >= len(sorted_sentence_idxes):
					break
			sum_lengths += doc
		return doc_ids




	# retrieve the top passages for each question. 
	def retrieve_passages(self,numPassages):
		for question in self.questions:
			print("Retrieving passages for Question: " + str(question[0]))
			all_sentences,doc_sentence_lengths = self.doc_sent_tokenizer(self.corpus[question[0]])

			self.doc_lengths[question[0]] = doc_sentence_lengths
			top_ten_words,indexes  = self.passage_retrieval2(question[1],all_sentences, numPassages)
			self.answers_ids[question[0]] = (top_ten_words,indexes.tolist())
		

	def tag_passages(self):
		for question in self.answers_ids:
			print('Tagging passages for Question '+str(question))
			self.tagged_passages[question] = []
			for passage in self.answers_ids[question][0]:
				tagged_passage = ''
				if self.taggedQuestions[question] == 'DATE':
					tagged_passage = self.dateTegger.tag(word_tokenize(passage))
				else:
					tagged_passage = self.nerTagger.tag(word_tokenize(passage))
				self.tagged_passages[question].append(tagged_passage)

			#filter out irrelevant passages
			listTags = []
			if self.taggedQuestions[question] == 'DATE':
				listTags.append('DATE')
			elif self.taggedQuestions[question] == 'PERSON':
				listTags.append('PERSON')
			else:
				listTags.append('LOCATION')
				listTags.append('ORGANIZATION')
			passagesToDelete = []
			for i in xrange(0,len(self.tagged_passages[question])):
				passage = self.tagged_passages[question][i]
				
				isRelevant = False
				for word in passage:
					if word[1] in listTags:
						isRelevant = True
				if not isRelevant:
					passagesToDelete.append(i)
			for i in xrange(len(passagesToDelete)-1,-1,-1):
				idx = passagesToDelete[i]
				del self.tagged_passages[question][idx]
				del self.answers_ids[question][0][idx]
				del self.answers_ids[question][1][idx]

			



	def compute_answers_NER(self):
		for question in self.answers_ids:
			answers = []
			indexes = []
			for i in xrange(0,min(5,len(self.answers_ids[question][1]))):
				tagged_passage = self.tagged_passages[question][i]
				answer = ''
				listTags = []
				if self.taggedQuestions[question] == 'DATE':
					listTags.append('DATE')
				elif self.taggedQuestions[question] == 'PERSON':
					listTags.append('PERSON')
				else:
					listTags.append('LOCATION')
					listTags.append('ORGANIZATION')
				print listTags
				foundAnswer = False
				scanning = True
				for word in tagged_passage:
					print word
					if not foundAnswer:
						if word[1] in listTags:
							foundAnswer	 = True
							answer = answer + word[0]
					else:
						if scanning:
							if word[1] in listTags:
								answer = answer +' ' +word[0]
							else:
								scanning =False
				
				answers.append(answer)
				
				idx = self.answers_ids[question][1][i]
				
				idxFinal = self.compute_idx(self.doc_lengths[question],idx)
				indexes.append(idxFinal)

			print answers
			print indexes
			if len(answers)<5:
				indexes = indexes + ([1] * (5-len(answers)))
				answers = answers + (["nil"] * (5-len(answers)))
				

			print answers
			print indexes
			
			self.finalAnswers[question] = answers
			self.finalIdxs[question] = indexes


		self.create_answers(self.finalIdxs.keys(),self.finalIdxs,self.finalAnswers)
			



	def compute_idx(self,doc_lengths,idx):
		running_sum = 0
		rel_lengths = []
		for length in doc_lengths:
			running_sum = running_sum+length
			rel_lengths.append(running_sum)
		if idx<=rel_lengths[0]:
			return 1
		elif rel_lengths[-2]<=idx:
			return len(doc_lengths)
		else:
			pos = 0
			while rel_lengths[pos]<=idx:
				pos = pos+1
			return pos+1




	# method that computes the best answers using only passage retrival for each question
	def compute_answers(self):
		question_ids = []
		doc_ids = {}
		
		for question in self.questions:
			print("Answering Question: " + str(question[0]))
			question_ids.append(question[0])
			all_sentences,doc_sentence_lengths = self.doc_sent_tokenizer(self.corpus[question[0]])
			top_ten_words,indexes  = self.passage_retrieval(question[1],all_sentences, 5)
			doc_ids[question[0]] = self.compute_document_id(indexes,doc_sentence_lengths)
			self.answers_ids[question[0]] = top_ten_words
		question_ids.sort()
		self.create_answers(question_ids,doc_ids,self.answers_ids)

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
			top_ten_words.append(' '.join(split[:10]))

		if len(top_ten_words)<5:
			top_ten_words = top_ten_words + (["nil"] * (5-len(top_ten_words)))
			indexes = indexes + ([1] * (5-len(top_ten_words)))

		return top_ten_words, indexes


	def passage_retrieval2(self, query, corpus, n_items):
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
			top_ten_words.append(' '.join(split))

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
	qa_system.tag_questions()
	qa_system.parse_documents()
	qa_system.retrieve_passages(10)
	qa_system.tag_passages()
	qa_system.compute_answers_NER()



if __name__ == '__main__':
	main()
