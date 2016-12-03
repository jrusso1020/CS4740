import gensim
import glob

class Sentences(object):
  def __init__(self, dirname):
    self.dirname = dirname

  def __iter__(self):
    for file_name in glob.glob(self.dirname + "/*.txt"):
      with open(file_name, 'r') as article:
        text = article.read()
        sents = text.split("\n\n")
        temp = ""
        for sent in sents:
          if sent != "":
            split = sent.split("\n")
            curr_sent = []
            for index, s in enumerate(split):
              tup = s.rsplit('\t', 1)
              if tup[1]=='_':
                curr_sent.append((tup[0] + " O").replace(" ", "/"))
                temp = "O"
              elif temp=="" and ('CUE' in tup[1]):

                curr_sent.append((tup[0] + " B-CUE").replace(" ", "/"))
                temp = "B-CUE"
              elif temp=="O" and ('CUE' in tup[1]):

                curr_sent.append((tup[0] + " B-CUE").replace(" ", "/"))
                temp = "B-CUE"
              elif temp=="B-CUE" and ('CUE' in tup[1]):

                curr_sent.append((tup[0] + " I-CUE").replace(" ", "/"))
                temp = "I-CUE"
              elif temp=="I-CUE" and ('CUE' in tup[1]):

                curr_sent.append((tup[0] + " I-CUE").replace(" ", "/"))
                temp = "I-CUE"
            yield curr_sent


class JustWords(object):
  def __init__(self, dirname):
    self.dirname = dirname

  def __iter__(self):
    for file_name in glob.glob(self.dirname + "/*.txt"):
      with open(file_name, 'r') as article:
        text = article.read()
        sents = text.split("\n\n")
        temp = ""
        for sent in sents:
          if sent != "":
            split = sent.split("\n")
            curr_sent = []
            for index, s in enumerate(split):
              tup = s.split('\t')
              curr_sent.append(tup[0])
            yield curr_sent


def main():
  sentences = Sentences("train")
  model = gensim.models.Word2Vec(sentences, min_count=1)
  print("FINISHED MAKING MODEL")

  # save the model
  model.save('word2vecmodel')

  just_words = JustWords("train")
  model2 = gensim.models.Word2Vec(just_words, min_count=1)
  print("FINISHED SECOND MODEL")
  model2.save('justwordsmodel')


if __name__ == '__main__':
  main()
