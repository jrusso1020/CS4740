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
        curr_sent = []
        for index, s in enumerate(split):
          tup = s.rsplit('\t', 1)
          if len(tup)>1:
            if len(curr_sent)==0:
              if tup[1]=='_':
                curr_sent.append((tup[0], "O"))
                temp = "O"
              elif 'CUE' in tup[1]:
                curr_sent.append((tup[0], "B-CUE"))
                temp = "B-CUE"
            else:
              if tup[1]=="_":
                curr_sent.append((tup[0], "O"))
                temp = "O"
              elif temp=="O" and ('CUE' in tup[1]):
                curr_sent.append((tup[0], "B-CUE"))
                temp = "B-CUE"
              elif temp=="B-CUE" and ('CUE' in tup[1]):
                curr_sent.append((tup[0], "I-CUE"))
                temp = "I-CUE"
              elif temp=="I-CUE" and ('CUE' in tup[1]):
                curr_sent.append((tup[0], "I-CUE"))
                temp = "I-CUE"
          if (".\t." in tup[0]) or ("!\t!" in tup[0]) or ("?\t?" in tup[0]):
            self.train_lines.append(curr_sent)
            curr_sent = []

  def parse_testing_files(self, directory):
    tagged_test = []
    for file_name in glob.glob(directory + "/*.txt"):
      with open(file_name, 'r') as article:
        text = article.read()
        split = text.split("\n")
        temp = ""
        curr_sent = []
        for index, s in enumerate(split):
          if s!="":
            curr_sent.append(s)
          if (".\t." in s) or ("!\t!" in s) or ("?\t?" in s):
            tagged_test.append(curr_sent)
            curr_sent = []
    return tagged_test

  def hmm_train(self):
    self.hmmtagger = self.hmmtrainer.train_supervised(self.train_lines)

  def hmm_predict(self, test_list):
    predicted = []
    for line in test_list:
      predicted.append(self.hmmtagger.tag(line))

    return predicted




def main():
  tagger = POStagger()
  tagger.parse_training_files("train")
  tagger.hmm_train()

  public = tagger.parse_testing_files("test-public")

  predicted = tagger.hmm_predict(public)



if __name__ == '__main__':
  main()
