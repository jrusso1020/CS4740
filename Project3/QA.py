import numpy as np
import nltk
import re
import os


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

		 



def main():
	qa_system = QASystem()
	qa_system.parse_questions('question.txt')
	qa_system.parse_documents()
	

if __name__ == '__main__':
	main()
