import sys
import nltk
import os
import numpy as np
import re
import math
from collections import Counter, defaultdict



class LMmodel():
  def __init__(self, dir_path):
    self.dir_path = dir_path
    self.start_token = "BEGIN"
    self.end_token = "END"
    self.unknown_token = "UNK"
    self.unigram_dist = None
    self.bigram_dist = None
    self.texts = []
    self.tokens = []
    self.modProbs = {}
    self.topic = ""
    self.testingTokens = []
    self.testingTexts = []

  # strip the header from the news articles
  def strip_newsgroup_header(self, text):
    #TODO: remove beginning email nonsense

  # Parse the corpus from the given directory
  # Remove beginnings of files
  # remove extraneous punction and add start and stop tokens

    if text.find("Subject :")!=-1:
      text = text[text.find("Subject :")+len("Subject :"):]



  # regex used for stripping non-necessary characters
    regex = re.compile('[^a-zA-Z.!? ]')
    puncRegex = re.compile('[!?]')
    parsedText= regex.sub("",text)
    parsedText = puncRegex.sub(".",parsedText)
    parsedText = parsedText.lower()

    return parsedText



  #split training data into training and test data for validation, and tokenize both sets at the same time 
  def parse_files_split(self,percentage):
    files = os.listdir(self.dir_path)
    trainFiles = list(files[0:round(percentage*len(files))])
    testFiles = list(files[round(percentage*len(files)):])
    for file in trainFiles:
      with open(self.dir_path + '/'+file, 'r') as article:
          string = article.read()
          parsedText = self.strip_newsgroup_header(string)
          self.texts.append(parsedText)
    self.tokenize()

    for file in testFiles:
      with open(self.dir_path + '/'+file, 'r') as article:
          string = article.read()
          parsedText = self.strip_newsgroup_header(string)
          self.testingTexts.append(parsedText)
    sentences = []
    for text in self.testingTexts:
        sentences = sentences + nltk.tokenize.sent_tokenize(text)
        

    for sentence in sentences:
      regex = re.compile('[.]')
      parsedSentence = self.start_token +" " +regex.sub(" "+self.end_token+" ",sentence)
      self.testingTokens = self.testingTokens + nltk.tokenize.word_tokenize(parsedSentence)



    





  def parse_files(self):
    for i in xrange (0,len(os.listdir(self.dir_path))):
      filename = os.listdir(self.dir_path)[i]
      #TODO check for .txt ""
      if True:
        with open(self.dir_path + '/'+filename, 'r') as article:
          string = article.read()

          #TODO: need to preprocess file, add start and end sentence markers, remove weird characters, remove weird start
          parsedText = self.strip_newsgroup_header(string)
          self.texts.append(parsedText)
        


  #tokenize the corpus 
  def tokenize(self):
    sentences = []
    for text in self.texts:
        sentences = sentences + nltk.tokenize.sent_tokenize(text)
        

    for sentence in sentences:
      regex = re.compile('[.]')
      parsedSentence = self.start_token +" " +regex.sub(" "+self.end_token+" ",sentence)
      self.tokens = self.tokens + nltk.tokenize.word_tokenize(parsedSentence)

  #compute the unsmoothed unigram probability distributions
  def unigram(self, tokens):
    total_tokens = len(tokens)
    unigram_freq = dict.fromkeys(tokens, 0)

    for token in tokens:
      unigram_freq[token] += 1

    prob_distribution = unigram_freq
    key_pairs = unigram_freq.items()

    for token, freq in key_pairs:
      prob_distribution[token] = float(freq) / float(total_tokens)

    self.unigram_dist = prob_distribution

  #compute the unsmoothed bigram probability distribution
  def bigram(self, tokens):
    total_tokens = Counter(tokens)

    #last token can't have a bigram
    total_tokens[self.end_token] -= 1

    bigram_freq = {token: defaultdict(int) for token in total_tokens}

    for i, token in enumerate(tokens[:-1]):
      bigram_freq[token][tokens[i+1]] += 1


    prob_distribution = bigram_freq
    key_pairs = bigram_freq.items()

    for token, freq_dict in key_pairs:
      freq_dict_key_pairs = freq_dict.items()
      for token_before, freq in freq_dict_key_pairs:
        freq_dict[token_before] = float(freq) / float(total_tokens[token])

    self.bigram_dist = prob_distribution

  #pick the next token for the generated sentence
  def pick_token(self, tokens, ngram_dist, ngram):
    if ngram == 1:
      keys = ngram_dist.keys()
      values = ngram_dist.values()
      

      token = np.random.choice(keys, p=values)
      while token == self.start_token:
        token = np.random.choice(keys,p=values)
      return token
    else:
      keys = ngram_dist[tokens[len(tokens) - 1]].keys()
      values = ngram_dist[tokens[len(tokens) - 1]].values()

      token = np.random.choice(keys, p=values)

      return token

  #generate a sentence using the type of ngram
  def sentence_generator(self, ngram):
    if ngram == 1:
      ngram_dist = self.unigram_dist
    else:
      ngram_dist = self.bigram_dist

    generated_sentence = []

    sentence_tokens = [self.start_token]

    token = self.pick_token(sentence_tokens, ngram_dist,ngram)
    
    while token != self.end_token:
      if ngram > 1:
        del sentence_tokens[0]
        sentence_tokens.append(token)
      generated_sentence.append(token)
      token = self.pick_token(sentence_tokens, ngram_dist, ngram)

    print(' '.join(generated_sentence))


    #compute the perplexity of our language model on a given test set using an n-gram model
  def perplexity(self, ngram,textTokens):
    runningSum = 0 
    if ngram == 1:
      for token in textTokens:
        runningSum = runningSum - math.log(self.unigram_dist[token])

      return math.exp(runningSum/len(textTokens))
    numTokens = 0 
    if ngram ==2:
      for i in xrange(0,len(textTokens)):
        token = textTokens[i]
        if token != self.start_token:
          if (textTokens[i-1],token) not in self.modProbs:
            runningSum = runningSum-math.log(self.modProbs[("","")])

          else:
            runningSum = runningSum - math.log(self.modProbs[(textTokens[i-1],token)])
          numTokens = numTokens + 1
      return math.exp(runningSum/numTokens)



  #Replace every word that occurs once with unknown symbol
  def replaceFirstSeen(self):
    wordFreq = Counter(self.tokens)
    
    for i in xrange(0,len(self.tokens)):
      token = self.tokens[i]
      if wordFreq[token] == 1:
        self.tokens[i] = self.unknown_token


  def goodTuring(self,T):
    #Data structure for bigram counts and bin counts 
    counts = {}
    countOfCounts = {}

    #Compute frequencey of each bigram
    for i in xrange(0,len(self.tokens)-1):
      key = (self.tokens[i],self.tokens[i+1])
      if key in counts:
        counts[key] = counts[key] + 1
      else:
        counts[key] = 1


    #compute N
    totalBigrams = 0
    for key in counts:
      totalBigrams = totalBigrams + counts[key]
   

    #Compute N_i for each possible i 
    for key in counts:
      if counts[key] in countOfCounts:
        countOfCounts[counts[key]] = countOfCounts[counts[key]] + 1
      else:
        countOfCounts[counts[key]] = 1

    #Compute the modified counts for each bigram
    modCounts = {}
    modProbs = {}
    wordCounts = Counter(self.tokens)
    for key in counts:
      c = counts[key]
      if  c<T:
        
        modCounts[key] = (c+1) * float(countOfCounts[c+1])/float(countOfCounts[c])
        
        #modCounts[key] = ((c + 1) *float(countOfCounts[c+1])/float(countOfCounts[c]) - c*(T+1) *countOfCounts[T+1]/countOfCounts[1])/(1-(T+1)*countOfCounts[T+1]/countOfCounts[1])

      else:
        modCounts[key] = counts[key]
      self.modProbs[key] = modCounts[key]/float(wordCounts[key[0]])
      self.modProbs[("","")] = float(countOfCounts[1])/(float(totalBigrams*((len(wordCounts)**2)-totalBigrams)))
      #print(abs(self.modProbs[key]-self.bigram_dist[key[0]][key[1]]))
      #print(self.bigram_dist[key[0]][key[1]])
      #print("ModProb")
      #print self.modProbs[key]





def tokenizeText(text):
  tokens = []
  if text.find("Subject :")!=-1:
      text = text[text.find("Subject :")+len("Subject :"):]

  sentences = nltk.tokenize.sent_tokenize(text)
  for sentence in sentences:
    regex = re.compile('[.]')
    parsedSentence = "BEGIN" +" " +regex.sub(" "+"END"+" ",sentence)
    tokens = tokens + nltk.tokenize.word_tokenize(parsedSentence)
  
  return tokens



#replace all words in a text that are not in a language model with unknown symbols
def removeUnseenWords(text,corpusTokens):

  for i in xrange(0,len(text)):
    token = text[i]    

    if token not in corpusTokens:
      text[i] = "UNK"










#Classify each test text by calculating the perplexity of each language model on it and selecting
#the topic with the least perplexity
def classify(languageModels):
  predictions = {}
  allPerplexities = {}
  test_files = 'data_corrected/classificationTask/test_for_classification'
  for i in xrange (0,len(os.listdir(test_files))):
      filename = os.listdir(test_files)[i]
      with open(test_files + '/'+filename, 'r') as article:
          string = article.read()
          tokens = tokenizeText(string)
          
          perplexities = []
          for lModel in languageModels:
            tokenCopy = list(tokens)
            removeUnseenWords(tokenCopy,lModel.tokens)
            perplexities.append(lModel.perplexity(2,tokenCopy))
          max_perplexity = min(perplexities)
          max_topic = perplexities.index(max_perplexity)
          predictions[filename] = languageModels[max_topic].topic
          allPerplexities[filename] = list(perplexities)
          print perplexities
          print("Predicted topic for " +filename +" is "+ predictions[filename])



  topicDict = {}
  
  topicDict['atheism'] = 0
  topicDict['autos'] =  1
  topicDict['computer_graphics']= 2
  topicDict['medicine'] = 3
  topicDict['motorcycles'] =  4
  topicDict['religion'] = 5
  topicDict['space'] =  6
  f = open('predictions.csv','w')
  f.write("Id,Prediction" + '\n')
  for file in predictions:
    f.write(file+","+str(topicDict[predictions[filename]])+'\n')



  print("Done with Predictions")





def main():
  topics = ["atheism",     "graphics"   , "motorcycles"   ,"space",
"autos",    "medicine",    "religion"]
  setOfModels = []
  for topic in topics:
    i = topics.index(topic)
    print("Training on "+topic)
    dir_path = 'data_corrected/classificationTask/'+topic+'/train_docs'
    setOfModels.append(LMmodel(dir_path))
    setOfModels[i].parse_files()
    setOfModels[i].tokenize()
    setOfModels[i].replaceFirstSeen()
    setOfModels[i].unigram(setOfModels[i].tokens)
    #setOfModels[topic].bigram(setOfModels[topic].tokens)
    setOfModels[i].goodTuring(0)
    setOfModels[i].topic = topic

  classify(setOfModels)







  








if __name__ == '__main__':
  main()
