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
  def perplexity(self, ngram):
    runningSum = 0 
    if ngram == 1:
      for token in self.tokens:
        runningSum = runningSum - math.log(self.unigram_dist[token])

      return math.exp(runningSum/len(self.tokens))
    numTokens = 0 
    if ngram ==2:
      for i in xrange(0,len(self.tokens)):
        token = self.tokens[i]
        if token != self.start_token:
          runningSum = runningSum - math.log(self.bigram_dist[self.tokens[i-1]][token])
          numTokens = numTokens + 1
      return math.exp(runningSum/numTokens)



  #Replace every first occurence of a token with the unknown symbol
  def replaceFirstSeen(self):
    seenWords = {}
    for i in xrange(0,len(self.tokens)):
      token = self.tokens[i]
      if token not in seenWords:
        seenWords[token] = 1
        self.tokens[i] = self.unknown_token




def main():
  dir_path = sys.argv[1]
  #nltk.download()
  model = LMmodel(dir_path)
  model.parse_files()
  model.tokenize()
  model.unigram(model.tokens)
  model.bigram(model.tokens)
  print model.perplexity(2)







if __name__ == '__main__':
  main()
