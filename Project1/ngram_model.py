import sys
import nltk
import os
from collections import Counter, defaultdict


class LMmodel():
  def __init__(self, dir_path):
    self.dir_path = dir_path
    self.start_token = "<<s>>"
    self.end_token = "<</s>>"
    self.unknown_token = "<<u>>"
    self.unigram_dist = None
    self.bigram_dist = None


  # Parse the corpus from the given directory
  # Remove beginnings of files
  # remove extraneous punction and add start and stop tokens
  def parse_files():
    for filename in os.listdir(dir_path):

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
      bigram_freq[token][tokens[i+1]] + = 1

    prob_distribution = bigram_freq
    key_pairs = bigram_freq.items()

    for token, freq_dict in key_pairs:
      freq_dict_key_pairs = freq_dict.items()
      for token_before, freq in freq_dict_key_pairs:
        freq_dict[token_before] = freq / total_tokens[token]

    self.bigram_dist = freq_dict

  #generate a sentence using the type of ngram
  def sentence_generator(self, ngram):


def main():
  dir_path = sys.argv[1]

  model = LMmodel(dir_path)




if __name__ == '__main__':
  main()