import sys
import nltk


class LMmodel():
  def __init__(self, file_path):
    self.file_path = file_path
    self.start_token = "<<s>>"
    self.end_token = "<</s>>"
    self.unknown_token = "<<u>>"


  def parse_file():


def main():
  file_path = sys.argv[1]

  model = LMmodel(file_path)




if __name__ == '__main__':
  main()