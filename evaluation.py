import random
import numpy as np
from tensorflow.keras.callbacks import Callback
from dataset import test_files, tokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
from dataset import load_files
from visualization import show_case

def compute_metrics(sentences, model, topk, maxlen):
    # 计算bleu、rouge-1、rouge-2、rouge-l指标
    rouge_1 = 0
    rouge_2 = 0
    rouge_l = 0
    bleu = 0
    total = 0
    for content, title in self.gen_sentences():
        total += 1
        pred_title = self.pred_title(content)
        scores = self.rouge.get_scores(hyps=pred_title, refs=title)
        rouge_1 += scores[0]["rouge-1"]["f"]
        rouge_2 += scores[0]["rouge-2"]["f"]
        rouge_l += scores[0]["rouge-l"]["f"]
        bleu += sentence_bleu(
            references=[title.split(" ")],
            hypothesis=pred_title.split(" "),
            smoothing_function=SmoothingFunction().method1
        )
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total
    result = {
        "rouge-1": rouge_1,
        "rouge-2": rouge_2,
        "rouge-l": rouge_l,
        "bleu": bleu,
    }
    return result

def load_test_sentences(files=test_files, k=12):
    # 从测试集中随机选择k个样本用于训练阶段的人工验证
    if k == -1:
        k = len(files)
    files = random.sample(test_files, k=k)
    return [(content, title) for content, title in load_files(files)]

def beam_search(model, token_ids, topk=3, maxlen=48):
    x_tokens = np.array([token_ids] * topk)
    y_tokens = np.array([[tokenizer.START]] * topk)
    scores = [0] * topk
    for i in range(maxlen):
        # 跳过MASK、UNK、START
        proba = model.predict([x_tokens, y_tokens])[:, -1, tokenizer.END:]
        log_proba = np.log(proba + 1e-12)
        # 选出topk
        arg_topk = log_proba.argsort(axis=1)[:,-topk:]
        temp_y_tokens = []
        temp_scores = []
        if i == 0:
            for j in range(topk):
                temp_y_tokens.append(list(y_tokens[j]) + [arg_topk[0][j]+3])
                temp_scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            # 遍历topk*topk的组合，并从中选择topk
            for j in range(topk):
                for k in range(topk):
                    temp_y_tokens.append(list(y_tokens[j]) + [arg_topk[j][k]+3])
                    temp_scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(temp_scores)[-topk:]
            temp_y_tokens = [temp_y_tokens[k] for k in _arg_topk]
            temp_scores = [temp_scores[k] for k in _arg_topk]
        y_tokens = np.array(temp_y_tokens)
        scores = np.array(temp_scores)
        best_one = np.argmax(scores)
        # 遇到END就直接跳出
        if y_tokens[best_one][-1] == tokenizer.END:
            return y_tokens[best_one]
    return y_tokens[np.argmax(scores)]

class Evaluator(Callback):

    def __init__(self, gen_sentences, tokenizer, file, maxlen=48, topk=3):
        self.gen_sentences = gen_sentences
        self.tokenizer = tokenizer
        self.file = file
        self.maxlen = maxlen # title maxlen
        self.topk = topk
        self.rouge = Rouge()
        self.min_loss = 1e12

    def on_epoch_end(self, epoch, logs=None):
        # 对比真实title与预测title
        print()
        self.human_check()
        val_loss = logs["val_loss"]
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.model.save_weights(self.file)

    def human_check(self):
        # 真实样例预测
        for content, title in self.gen_sentences():
            ptitle = self.predict_title(content)
            show_case(content, title, ptitle)

    def predict_title(self, content, topk=3, maxlen=48):
        content_ids = self.tokenizer.encode(content)
        title_ids = beam_search(self.model, content_ids, self.topk, self.maxlen)
        title = self.tokenizer.decode(title_ids)
        return title
