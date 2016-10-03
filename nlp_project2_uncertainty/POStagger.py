import glob
import sys
import nltk
import numpy as np

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
        sents = text.split("\n\n")
        temp = ""
        for sent in sents:
          if sent != "":
            split = sent.split("\n")
            curr_sent = []
            for index, s in enumerate(split):
              tup = s.rsplit('\t', 1)
              if len(tup) > 1:
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
            self.train_lines.append(curr_sent)


  # parse the testing directory
  def parse_testing_files(self, directory):
    tagged_test = []
    for file_name in glob.glob(directory + "/*.txt"):
      with open(file_name, 'r') as article:
        text = article.read()
        sents = text.split("\n\n")
        for sent in sents:
          if sent != "":
            split = sent.split("\n")
            curr_sent = []
            for index, s in enumerate(split):
              if s!= "":
                curr_sent.append(s)
            tagged_test.append(curr_sent)
    return tagged_test

  # using preknown hedgewords to predict uncertainty
  def build_hedge_dict(self):
    self.baseline_dictionary["about"] = "B-CUE"
    self.baseline_dictionary["apparently"] = "B-CUE"
    self.baseline_dictionary["appear"] = "B-CUE"
    self.baseline_dictionary["around"] = "B-CUE"
    self.baseline_dictionary["basically"] = "B-CUE"
    self.baseline_dictionary["can"] = "B-CUE"
    self.baseline_dictionary["could"] = "B-CUE"
    self.baseline_dictionary["effectively"] = "B-CUE"
    self.baseline_dictionary["evidently"] = "B-CUE"
    self.baseline_dictionary["fairly"] = "B-CUE"
    self.baseline_dictionary["generally"] = "B-CUE"
    self.baseline_dictionary["hopefully"] = "B-CUE"
    self.baseline_dictionary["largely"] = "B-CUE"
    self.baseline_dictionary["likely"] = "B-CUE"
    self.baseline_dictionary["mainly"] = "B-CUE"
    self.baseline_dictionary["may"] = "B-CUE"
    self.baseline_dictionary["maybe"] = "B-CUE"
    self.baseline_dictionary["mostly"] = "B-CUE"
    self.baseline_dictionary["overall"] = "B-CUE"
    self.baseline_dictionary["perhaps"] = "B-CUE"
    self.baseline_dictionary["presumably"] = "B-CUE"
    self.baseline_dictionary["pretty"] = "B-CUE"
    self.baseline_dictionary["probably"] = "B-CUE"
    self.baseline_dictionary["clearly"] = "B-CUE"
    self.baseline_dictionary["quite"] = "B-CUE"
    self.baseline_dictionary["rather"] = "B-CUE"
    self.baseline_dictionary["really"] = "B-CUE"
    self.baseline_dictionary["seem"] = "B-CUE"
    self.baseline_dictionary["somewhat"] = "B-CUE"
    self.baseline_dictionary["supposedly"] = "B-CUE"

  # build a baseline uncertainty dictionary where all uncertain tokens are in it
  def build_baseline_dict(self):
    for sentence in self.train_lines:
      for tup in sentence:
        if ("CUE" in tup[1]) and (tup[0] not in self.baseline_dictionary):
          self.baseline_dictionary[tup[0]] = tup[1]

  # predict using hedge word dictionary
  def predict_hedge_baseline(self, test_list):
    predicted = []
    for line in test_list:
      curr_sent = []
      for token in line:
        if token.split("\t")[0] in self.baseline_dictionary:
          curr_sent.append((token, "B-CUE"))
        else:
          curr_sent.append((token, "O"))
      predicted.append(curr_sent)

    return predicted

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

  # post processer on the sentence level for determing which sentences have uncertainty
  def sentence_post_processing(self, test, predicted):
    string = "SENTENCE-" + test + ","
    for idx, sentence in enumerate(predicted):
      for tup in sentence:
        if tup[1]=="B-CUE" or tup[1]=="I-CUE":
          string += str(idx) + " "
          break

    return string

  #calculate the span ranges in the predictions for kaggle
  def spanRanges(self,predictions):
    ranges = []
    inSpan = False
    curRange = [0,0]
    counter = 0
    for i in xrange(0,len(predictions)):
      sentence = predictions[i]
      for j in xrange(0,len(sentence)):
        tup = sentence[j]
        if inSpan and tup[1] != "I-CUE":
          curRange[1] = counter -1
          inSpan = False
          ranges.append((curRange[0],curRange[1]))
        if not inSpan and tup[1] =="B-CUE":
          inSpan = True
          curRange[0] = counter
        counter = counter+1
    formattedRanges = [str(tup[0])+'-'+str(tup[1]) for tup in ranges]
    return " ".join(formattedRanges)




def main():
  tagger = POStagger()
  tagger.parse_training_files("train")
  tagger.hmm_train()

  tagger.build_hedge_dict()

  public = tagger.parse_testing_files("test-public")
  private = tagger.parse_testing_files("test-private")

  baseline_private_predicted = tagger.predict_hedge_baseline(private)

  baseline_public_predicted = tagger.predict_hedge_baseline(public)

  rangesPublic = tagger.spanRanges(baseline_public_predicted)
  rangesPrivate = tagger.spanRanges(baseline_private_predicted)

  s_public = tagger.sentence_post_processing("public", baseline_public_predicted)

  s_private = tagger.sentence_post_processing("private", baseline_private_predicted)

  csv = "Type,Indices\n"
  csv += s_public + "\n"
  csv += s_private + "\n"
  baseline = open("baseline_sentence.csv", 'w')
  baseline.write(csv)
  baseline.close()

  #predicted = tagger.hmm_predict(public)
  predicted
  #tagger.spanRanges(predicted)
  #s = tagger.sentence_post_processing("public", predicted)



if __name__ == '__main__':
  main()
