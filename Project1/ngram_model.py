import sys
import nltk
import os
import numpy as np
from collections import Counter, defaultdict



class LMmodel():
  def __init__(self, dir_path):
    self.dir_path = dir_path
    self.start_token = "<<s>>"
    self.end_token = "<</s>>"
    self.unknown_token = "<<u>>"
    self.unigram_dist = None
    self.bigram_dist = None

  # strip the header from the news articles
  def strip_newsgroup_header(self, text):
    #TODO: remove beginning email nonsense

  # Parse the corpus from the given directory
  # Remove beginnings of files
  # remove extraneous punction and add start and stop tokens
  def parse_files(self):
    for filename in os.listdir(self.dir_path):
      if filename.endswith(".txt"):
        with open(self.dir_path + filename, 'r') as article:
          string = article.read()
          #TODO: need to preprocess file, add start and end sentence markers, remove weird characters, remove weird start
        break


  #compute the unsmoothed unigram probability distributions
  def unigram(self, tokens):
    total_tokens = len(tokens)
    unigram_freq = dict.fromkeys(tokens, 0)

    for token in tokens:
      unigram_freq[token] += 1

    prob_distribution = unigram_freq
    key_pairs = unigram_freq.items()

    for token, freq in key_pairs:
      prob_distribution[token] = freq / total_tokens

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
        freq_dict[token_before] = freq / total_tokens[token]

    self.bigram_dist = freq_dict

  #pick the next token for the generated sentence
  def pick_token(self, tokens, ngram_dist, ngram):
    if ngram == 1:
      keys = ngram_dist.keys()
      values = ngram_dist.values()

      token = np.random.choice(keys, p=values)

      return token
    else:
      keys = ngram_dist[sentence_tokens[len(sentence_tokens) - 1]].keys()
      values = ngram_dist[sentence_tokens[len(sentence_tokens) - 1]].values()

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

    token = self.pick_token(sentence_tokens, ngram_dist)
    while token != self.end_token:
      if ngram > 1:
        del sentence_tokens[0]
        sentence_tokens.append(token)
      generated_sentence.append(token)
      word = self.pick_token(sentence_tokens, ngram_dist)

    print(' '.join(generated_sentence))


def main():
  dir_path = sys.argv[1]

  model = LMmodel(dir_path)
  model.parse_files()




if __name__ == '__main__':
  main()
