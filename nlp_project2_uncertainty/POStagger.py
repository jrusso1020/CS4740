import glob
import sys
import nltk

class POStagger():
  def __init__(self):
    self.train_lines = []
    self.hmmtrainer = nltk.tag.HiddenMarkovModelTrainer()
    self.hmmtagger = None
    self.baseline_dictionary = {}

  # parse the training directory putting BIO CUE's for all CUES
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

  # parse the testing directory
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

  # build a baseline uncertainty dictionary where all uncertain tokens are in it
  def build_baseline_dict(self):
    for sentence in self.train_lines:
      for tup in sentence:
        if ("CUE" in tup[1]) and (tup[0] not in self.baseline_dictionary):
          self.baseline_dictionary[tup[0]] = tup[1]

  # predict uncertainty based on baseline dictionary occurrences
  def predict_baseline(self, test_list):
    predicted = []
    for line in test_list:
      curr_sent = []
      for token in line:
        if token in self.baseline_dictionary:
          curr_sent.append((token, "B-CUE"))
        else:
          curr_sent.append((token, "O"))
      predicted.append(curr_sent)

    return predicted

  # wrapper function for the HMM training method provided by nltk
  def hmm_train(self):
    self.hmmtagger = self.hmmtrainer.train_supervised(self.train_lines)

  # wrapper function for the HMM tagging method provided by nltk
  def hmm_predict(self, test_list):
    predicted = []
    for line in test_list:
      predicted.append(self.hmmtagger.tag(line))

    return predicted




def main():
  tagger = POStagger()
  tagger.parse_training_files("train")
  tagger.hmm_train()

  tagger.build_baseline_dict()

  public = tagger.parse_testing_files("test-public")

  baseline_public_predicted = tagger.predict_baseline(public)

  predicted = tagger.hmm_predict(public)



if __name__ == '__main__':
  main()
