from dataset import load_files, train_files
from textcolor import *

for content, title in load_files(train_files):
    print_match_subsequence(content, title)
    input()
