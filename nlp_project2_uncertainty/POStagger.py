import glob
import sys
import nltk
import numpy as np

class POStagger():
  def __init__(self):
    self.train_lines = []
    self.hmmtrainer = nltk.tag.HiddenMarkovModelTrainer()
    self.hmmtagger = None
    self.crftagger = nltk.tag.CRFTagger(feature_func = self.get_features)
    self.perceptrontagger = nltk.tag.perceptron.PerceptronTagger(load=False)
    self.baseline_dictionary = {}
    self.val_train_lines = []
    self.val_test_lines_answers = []
    self.val_test_lines = []

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
              if tup[1]=='_':
                curr_sent.append((tup[0], "O"))
                temp = "O"
              elif temp=="" and ('CUE' in tup[1]):

                curr_sent.append((tup[0], "B-CUE"))
                temp = "B-CUE"
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
        if ("CUE" in tup[1]) and (tup[0].split("\t")[0] not in self.baseline_dictionary):
          self.baseline_dictionary[tup[0].split("\t")[0]] = "B-CUE"

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
    self.hmmtagger = self.hmmtrainer.train_supervised(self.val_train_lines)

  # wrapper function for the HMM tagging method provided by nltk
  def hmm_predict(self, test_list):
    predicted = []
    for line in test_list:
      predicted.append(self.hmmtagger.tag(line))

    return predicted

  # wrapper function to train the CRF tagger provided by nltk
  def crf_train(self):
    self.crftagger.train(self.val_train_lines, 'model.crf.tagger')

  # wrapper function to tag using the CRF tagger provided by nltk
  def crf_predict(self, test_list):
    return self.crftagger.tag_sents(test_list)

  # wrapper function to train the perceptron tagger provided by nltk
  def perceptron_train(self,nr_iter = 1000):
    self.perceptrontagger.train(self.val_train_lines)

  # wrapper function to tag using the perceptron tagger provided by nltk
  def perceptron_predict(self, test_list):
    predicted = []
    for line in test_list:
      predicted.append(self.perceptrontagger.tag(line))

    return predicted

  def get_features_simple(self, tokens, idx):
    features = []
    features.append(tokens[idx].split()[0])
    features.append(tokens[idx].split()[1])
    return features

  def get_word_features(self,tokens,idx):
    features = []
    rangeLength = 2
    features.append(tokens[idx].split()[0])
    for i in range(idx-rangeLength,idx+rangeLength):
      features.append(tokens[i%len(tokens)].split()[1])

    return features

  def get_features_1(self,tokens,idx):
    features = []
    features.append(tokens[idx].split()[0])
    features.append(tokens[idx].split()[1])
    if idx>0:
      features.append(tokens[idx-1].split()[0])
      features.append(tokens[idx-1].split()[1])
    if idx<len(tokens)-1:
      features.append(tokens[idx+1].split()[0])
      features.append(tokens[idx+1].split()[1])
    return features

  def get_features(self,tokens,idx):
    rangeLength = 2
    features = []

    for i in range(idx-rangeLength,idx+rangeLength):
      features.append(tokens[i%len(tokens)].split()[0])
      features.append(tokens[i%len(tokens)].split()[1])


    return features

  # post processer on the sentence level for determing which sentences have uncertainty
  def sentence_post_processing(self, test, predicted):
    #string = "SENTENCE-" + test + ","

    string = ""
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
    for i in range(len(predictions)):
      sentence = predictions[i]
      for j in range(len(sentence)):
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

  # split the training data into 75% training and 25% testing
  def split_training(self):
    self.val_train_lines = self.train_lines[:int(len(self.train_lines) * 0.75)]

    self.val_test_lines_answers = self.train_lines[int(len(self.train_lines) * 0.75):]

    for list_tups in self.val_test_lines_answers:
      first_items = [str(i[0]) for i in list_tups]
      self.val_test_lines.append(first_items)

  # helper to break up our string of spans into a list of numbers
  def break_up_spans(self, string_spans):
    s_split = string_spans.split(" ")

    span_list = []
    for x in s_split:
      temp = x.split("-")
      if len(temp)>1:
        span_list = span_list + list(range(int(temp[0]), int(temp[1]) + 1))

    return span_list

  # calculate the precision of approach using our val_test_lines prediction vs the actual
  def precision(self, sent, actual, predicted):
    act = []
    pred = []
    if sent:
      act = actual.split(" ")
      del act[-1]
      pred = predicted.split(" ")
      del pred[-1]
    else:
      act = self.break_up_spans(actual)
      pred = self.break_up_spans(predicted)


    num_correct = len(set(act) & set(pred))
    pred_pos = len(pred)

    return num_correct / pred_pos

  # calculate the recall of approach using our val_test_lines prediction vs the actual
  def recall(self, sent, actual, predicted):
    act = []
    pred = []
    if sent:
      act = actual.split(" ")
      del act[-1]
      pred = predicted.split(" ")
      del pred[-1]
    else:
      act = self.break_up_spans(actual)
      pred = self.break_up_spans(predicted)

    num_correct = len(set(act) & set(pred))
    act_pos = len(act)

    return num_correct / act_pos

  # calculate the f measure of our approach from the precision and recall
  def f_measure(self, prec, recall):
    return 2 * ((prec * recall)/ (prec + recall))



#baseline dictionary tagger results
def dictExperiment(tagger,test_answers_sent,test_answers_span):
  dict_test = tagger.predict_hedge_baseline(tagger.val_test_lines)
  #print(dict_test)
  dict_test_ranges = tagger.spanRanges(dict_test)
  dict_test_sent = tagger.sentence_post_processing("",dict_test)
  dict_range_p = tagger.precision(False, test_answers_span, dict_test_ranges)
  dict_range_r = tagger.recall(False, test_answers_span, dict_test_ranges)
  dict_range_f = tagger.f_measure(dict_range_p, dict_range_r)

  print("Dictionary Span Range")
  print("Precision: " + str(dict_range_p))
  print("Recall: " + str(dict_range_r))
  print("F-Measure: " + str(dict_range_f))


  dict_sent_p = tagger.precision(True, test_answers_sent, dict_test_sent)
  dict_sent_r = tagger.recall(True, test_answers_sent, dict_test_sent)
  dict_sent_f = tagger.f_measure(dict_sent_p, dict_sent_r)

  print("Dict Sentences")
  print("Precision: " + str(dict_sent_p))
  print("Recall: " + str(dict_sent_r))
  print("F-Measure: " + str(dict_sent_f))



# split training data and test out different taggers for validation
def experiments():
  tagger = POStagger()
  tagger.parse_training_files("train")
  tagger.split_training()
  tagger.build_baseline_dict()
  tagger.hmm_train()
  tagger.crf_train()
  tagger.perceptron_train()


  # self testing results


  hmm_test = tagger.hmm_predict(tagger.val_test_lines)

  crf_test = tagger.crf_predict(tagger.val_test_lines)

  perc_test = tagger.perceptron_predict(tagger.val_test_lines)



  hmm_test_ranges = tagger.spanRanges(hmm_test)
  hmm_test_sent = tagger.sentence_post_processing("", hmm_test)

  crf_test_ranges = tagger.spanRanges(crf_test)
  crf_test_sent = tagger.sentence_post_processing("", crf_test)


  perc_test_ranges = tagger.spanRanges(perc_test)
  perc_test_sent = tagger.sentence_post_processing("", perc_test)

  test_answers_span = tagger.spanRanges(tagger.val_test_lines_answers)
  test_answers_sent = tagger.sentence_post_processing("", tagger.val_test_lines_answers)


  dictExperiment(tagger,test_answers_sent,test_answers_span)



  hmm_range_p = tagger.precision(False, test_answers_span, hmm_test_ranges)
  hmm_range_r = tagger.recall(False, test_answers_span, hmm_test_ranges)
  hmm_range_f = tagger.f_measure(hmm_range_p, hmm_range_r)

  print("HMM Span Range")
  print("Precision: " + str(hmm_range_p))
  print("Recall: " + str(hmm_range_r))
  print("F-Measure: " + str(hmm_range_f))

  hmm_sent_p = tagger.precision(True, test_answers_sent, hmm_test_sent)
  hmm_sent_r = tagger.recall(True, test_answers_sent, hmm_test_sent)
  hmm_sent_f = tagger.f_measure(hmm_sent_p, hmm_sent_r)

  print("HMM Sentences")
  print("Precision: " + str(hmm_sent_p))
  print("Recall: " + str(hmm_sent_r))
  print("F-Measure: " + str(hmm_sent_f))

  crf_range_p = tagger.precision(False, test_answers_span, crf_test_ranges)
  crf_range_r = tagger.recall(False, test_answers_span, crf_test_ranges)
  crf_range_f = tagger.f_measure(crf_range_p, crf_range_r)

  print("CRF Span Range")
  print("Precision: " + str(crf_range_p))
  print("Recall: " + str(crf_range_r))
  print("F-Measure: " + str(crf_range_f))

  crf_sent_p = tagger.precision(True, test_answers_sent, crf_test_sent)
  crf_sent_r = tagger.recall(True, test_answers_sent, crf_test_sent)
  crf_sent_f = tagger.f_measure(crf_sent_p, hmm_sent_r)

  print("CRF Sentences")
  print("Precision: " + str(crf_sent_p))
  print("Recall: " + str(crf_sent_r))
  print("F-Measure: " + str(crf_sent_f))

  perc_range_p = tagger.precision(False, test_answers_span, perc_test_ranges)
  perc_range_r = tagger.recall(False, test_answers_span, perc_test_ranges)
  perc_range_f = tagger.f_measure(perc_range_p, perc_range_r)

  print("Average Perceptron Span Range")
  print("Precision: " + str(perc_range_p))
  print("Recall: " + str(perc_range_r))
  print("F-Measure: " + str(perc_range_f))

  perc_sent_p = tagger.precision(True, test_answers_sent, perc_test_sent)
  perc_sent_r = tagger.recall(True, test_answers_sent, perc_test_sent)
  perc_sent_f = tagger.f_measure(perc_sent_p, perc_sent_r)

  print("Average Perceptron Sentences")
  print("Precision: " + str(perc_sent_p))
  print("Recall: " + str(perc_sent_r))
  print("F-Measure: " + str(perc_sent_f))

def main():
  experiments()
  # tagger = POStagger()
  # tagger.parse_training_files("train")
  # tagger.crf_train()



  # # tagger.build_hedge_dict()

  # public = tagger.parse_testing_files("test-public")
  # private = tagger.parse_testing_files("test-private")

  # baseline_private_predicted = tagger.predict_hedge_baseline(private)

  # baseline_public_predicted = tagger.predict_hedge_baseline(public)

  # rangesPublic = tagger.spanRanges(baseline_public_predicted)
  # rangesPrivate = tagger.spanRanges(baseline_private_predicted)

  # s_public = tagger.sentence_post_processing("public", baseline_public_predicted)

  # s_private = tagger.sentence_post_processing("private", baseline_private_predicted)

  # hmm_public = tagger.hmm_predict(public)
  # hmm_private = tagger.hmm_predict(private)

  # hmm_pub_ranges = tagger.spanRanges(hmm_public)
  # hmm_pub_sentences = tagger.sentence_post_processing("public", hmm_public)

  # hmm_priv_ranges = tagger.spanRanges(hmm_private)
  # hmm_priv_sentences = tagger.sentence_post_processing("private", hmm_private)


  # csv = "Type,Indices\n"
  # csv += hmm_pub_sentences + "\n"
  # csv += hmm_priv_sentences
  # baseline = open("hmm_sentence.csv", 'w')
  # baseline.write(csv)
  # baseline.close()

  # csv = "Type,Spans\n"
  # csv += "CUE-public," + hmm_pub_ranges + "\n"
  # csv += "CUE-private," + hmm_priv_ranges
  # baseline = open("hmm_span.csv", 'w')
  # baseline.write(csv)
  # baseline.close()

  # crf_public = tagger.crf_predict(public)
  # crf_private = tagger.crf_predict(private)

  # crf_pub_ranges = tagger.spanRanges(crf_public)
  # crf_pub_sentences = tagger.sentence_post_processing("public", crf_public)

  # crf_priv_ranges = tagger.spanRanges(crf_private)
  # crf_priv_sentences = tagger.sentence_post_processing("private", crf_private)


  # csv = "Type,Indices\n"
  # csv += crf_pub_sentences + "\n"
  # csv += crf_priv_sentences
  # baseline = open("crf_sentenceBest.csv", 'w')
  # baseline.write(csv)
  # baseline.close()

  # csv = "Type,Spans\n"
  # csv += "CUE-public," + crf_pub_ranges + "\n"
  # csv += "CUE-private," + crf_priv_ranges
  # baseline = open("crf_spanBest.csv", 'w')
  # baseline.write(csv)
  # baseline.close()

  # perc_public = tagger.perceptron_predict(public)
  # perc_private = tagger.perceptron_predict(private)

  # perc_pub_ranges = tagger.spanRanges(perc_public)
  # perc_pub_sentences = tagger.sentence_post_processing("public", perc_public)

  # perc_priv_ranges = tagger.spanRanges(perc_private)
  # perc_priv_sentences = tagger.sentence_post_processing("private", perc_private)


  # csv = "Type,Indices\n"
  # csv += perc_pub_sentences + "\n"
  # csv += perc_priv_sentences
  # baseline = open("perc_sentence.csv", 'w')
  # baseline.write(csv)
  # baseline.close()

  # csv = "Type,Spans\n"
  # csv += "CUE-public," + perc_pub_ranges + "\n"
  # csv += "CUE-private," + perc_priv_ranges
  # baseline = open("perc_span.csv", 'w')
  # baseline.write(csv)
  # baseline.close()










if __name__ == '__main__':
  main()