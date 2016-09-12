import sys
import nltk
import os


class LMmodel():
  def __init__(self, dir_path):
    self.dir_path = dir_path
    self.start_token = "<<s>>"
    self.end_token = "<</s>>"
    self.unknown_token = "<<u>>"


  def parse_file():


def main():
  dir_path = sys.argv[1]

  model = LMmodel(dir_path)




if __name__ == '__main__':
  main()