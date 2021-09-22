import argparse

from train_model import model
from dataset import tokenizer
from evaluation import beam_search
from textcolor import print_match_subsequence
from visualization import show_case

def predict_title(content, topk, maxlen):
    content_ids = tokenizer.encode(content)
    title_ids = beam_search(model, content_ids, topk, maxlen)
    title = tokenizer.decode(title_ids)
    return title

def show_prediction(content, title, pred_title):
    print("真实标题与内容对比：")
    print_match_subsequence(title, content)
    print()
    print("真实title：", title)
    print("预测title：", pred_title)
    print("=" * 20)

if __name__ == "__main__":
    import random
    from dataset import test_files as files
    from dataset import load_files

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="", help="content file")
    parser.add_argument("--content", type=str, default="")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--maxlen", type=int, default=48)
    args = parser.parse_args()

    file = args.file
    content = args.content
    if file == "" and content == "":
        random.shuffle(files)
        for content, title in load_files(files):
            if len(content) > 32 * 12:
                continue
            pred_title = predict_title(content, args.topk, args.maxlen)
            # show_prediction(content, title, pred_title)
            show_case(content, title, pred_title)
            input() # <Enter>预测下一个样本
    else:
        if content == "":
            with open(file, "r") as fd:
                content = fd.read()
        pred_title = predict_title(content, args.topk, args.maxlen)
        print_match_subsequence(content, pred_title)
