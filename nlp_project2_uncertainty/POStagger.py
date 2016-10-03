import glob
import sys
import nltk

class POStagger():
  def __init__(self):
    self.train_lines = []
    self.hmmtrainer = nltk.tag.HiddenMarkovModelTrainer()
    self.hmmtagger = None

  def parse_training_files(self, directory):
    for file_name in glob.glob(directory + "/*.txt"):
      with open(file_name, 'r') as article:
        text = article.read()
        split = text.split("\n")
        temp = ""
        for index, s in enumerate(split):
          tup = s.rsplit('\t', 1)
          if len(tup)>1:
            if len(self.train_lines)==0:
              if tup[1]=='_':
                self.train_lines.append((tup[0], "O"))
                temp = "O"
              elif 'CUE' in tup[1]:
                self.train_lines.append((tup[0], "B-CUE"))
                temp = "B-CUE"
            else:
              if tup[1]=="_":
                self.train_lines.append((tup[0], "O"))
                temp = "O"
              elif temp=="O" and ('CUE' in tup[1]):
                self.train_lines.append((tup[0], "B-CUE"))
                temp = "B-CUE"
              elif temp=="B-CUE" and ('CUE' in tup[1]):
                self.train_lines.append((tup[0], "I-CUE"))
                temp = "I-CUE"
              elif temp=="I-CUE" and ('CUE' in tup[1]):
                self.train_lines.append((tup[0], "I-CUE"))
                temp = "I-CUE"

  def parse_testing_files(self, directory):
    tagged_train = []
    for file_name in glob.glob(directory + "/*.txt"):
      with open(file_name, 'r') as article:
        text = article.read()
        split = text.split("\n")
        temp = ""
        for index, s in enumerate(split):
          if s!="":
            tagged_train.append(split)

    return tagged_train

  def hmm_train(self):
    self.hmmtagger = hmmtrainer.train_supervised(self.train_lines)




def main():
  tagger = POStagger()
  tagger.parse_training_files("train")


if __name__ == '__main__':
  main()
