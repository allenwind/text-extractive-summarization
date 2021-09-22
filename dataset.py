import os
import json
import glob
import collections
import random
import re
import numpy as np
import tensorflow as tf

_THUCNews = "/home/zhiwen/workspace/dataset/THUCTC/THUCNews/**/*.txt"
def list_thucnews_files(path=_THUCNews):
    """可到 http://thuctc.thunlp.org/ 下载"""
    files = glob.glob(path)
    files.sort()
    return files

def load_files(files):
    for path in files:
        with open(path, encoding="utf-8") as fd:
            title = fd.readline().strip()
            title = process_title(title)
            content = fd.read().strip()
            content = process_content(content)
        yield content, title

def train_val_test_split(files, train_size=0.8, val_size=0.1, test_size=0.1, shuffle=True, seed=782):
    # 文件级别的交叉验证
    if shuffle:
        random.Random(seed).shuffle(files)
        random.Random(seed ** 2).shuffle(files)
    length = len(files)
    i = int(length * train_size)
    j = int(length * (train_size + val_size))
    train_files = files[:i]
    val_files = files[i:j]
    test_files = files[j:]
    return train_files, val_files, test_files

def textQ2B(text):
    # 把文本从全角符号转半角符号
    rtext = []
    for c in text:
        c = ord(c)
        # \u3000
        if c == 12288:
            c = 32
        # 全角字符
        elif (c >= 65281 and c <= 65374):
            c -= 65248
        rtext.append(chr(c))
    return "".join(rtext)

def process_content(content):
    # content的预处理
    content = content.replace("\n", "。") \
                     .replace("\xa0", "")
    content = content.lower()
    content = textQ2B(content)
    return content

def process_title(title):
    # 标题的预处理
    title = re.sub("\(.+?\)", lambda x:"", title)
    title = title.replace(" ", ",")
    title = title.lower()
    title = textQ2B(title)
    return title

class Counter(dict):
    """计数器"""

    def __missing__(self, key):
        return 0

class Tokenizer:

    def __init__(self, maxlen=512):
        self.chars = None
        self.id2char = None
        self.char2id = None
        self.tags = {0:"MASK", 1:"UNK", 2:"START", 3:"END"}
        self.MASK = 0
        self.UNK = 1
        self.START = 2
        self.END =  3
        self.maxlen = maxlen # content最大长度
        self.filters = set("∵∴!\"#$%&'()[]*+,-./，。！@·……（）【】<>《》?？；‘’“”")

    def fit(self, files, minfreq=48):
        chars = Counter()
        for content, title in load_files(files):
            for c in content:
                chars[c] += 1
            for c in title:
                chars[c] += 1
        self.chars = {i:j for i,j in chars.items() if j > minfreq}
        self.id2char = {i:j for i,j in enumerate(self.chars, start=4)} # 4tags
        self.char2id = {j:i for i,j in self.id2char.items()}

    def fit_in_parallel(self, files, minfreq=32):
        pass

    def encode(self, text, with_tags=False, lower_case=True):
        # text转换为tokens
        if lower_case:
            text = text.lower()
        if with_tags:
            gen = (self.char2id.get(char, self.UNK) for char in text[:self.maxlen-2])
            tokens = [self.START, *gen, self.END]
        else:
            tokens = [self.char2id.get(char, self.UNK) for char in self.truncated_text(text, self.maxlen)]
        return tokens

    def decode(self, tokens):
        # tokens转换为text
        return "".join([self.id2char.get(i, "") for i in tokens])

    def truncated_text(self, text, maxlen):
        # 首尾截断
        length = len(text)
        if length <= maxlen:
            return text

        x = maxlen // 2
        text = text[:x] + text[-(maxlen - x):]
        return text

    @property
    def vocab_size(self):
        # 4 tags
        return len(self.chars) + len(self.tags)

    def __len__(self):
        return vocab_size

    def save(self, file):
        with open(file, "w") as fp:
            json.dump(
                [self.chars, self.id2char, self.char2id],
                fp,
                indent=4,
                ensure_ascii=False
            )

    def load(self, file):
        with open(file, "r") as fp:
            chars, id2char, char2id = json.load(fp)
        self.chars = chars
        self.id2char = {int(i):j for i,j in id2char.items()}
        self.char2id = char2id

class DataGenerator:

    def __init__(self, files, epochs, tokenizer):
        self.files = files
        self.epochs = epochs
        self.tokenizer = tokenizer

    def __call__(self):
        for _ in range(self.epochs):
            random.shuffle(self.files)
            for content, title in load_files(self.files):
                content = self.tokenizer.encode(content, with_tags=False)
                title = self.tokenizer.encode(title, with_tags=True)
                yield (content, title), ()

def create_dataset(files, epochs, batch_size, tokenizer, drop_remainder=True):
    dl = tf.data.Dataset.from_generator(
        generator=DataGenerator(files, epochs, tokenizer),
        output_types=((tf.int32, tf.int32), ())
    ).padded_batch(
        batch_size=batch_size,
        padded_shapes=(([None], [None]), ()),
        drop_remainder=drop_remainder
    # prefetch加速训练
    ).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )
    return dl

files = list_thucnews_files()
train_files, val_files, test_files = train_val_test_split(
    files=files,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1
)

tokenizer = Tokenizer()
file = "weights/tokens.json"
if os.path.exists(file):
    tokenizer.load(file)
else:
    print("tokenize...")
    tokenizer.fit(files)
    tokenizer.save(file)

dataset = create_dataset(train_files, epochs=100, batch_size=128, tokenizer=tokenizer)
dataset_val = create_dataset(val_files, epochs=1, batch_size=128, tokenizer=tokenizer)
dataset_test = create_dataset(test_files, epochs=1, batch_size=128, tokenizer=tokenizer)
vocab_size = tokenizer.vocab_size

# dataset = tf.data.Dataset.from_tensor_slices(train_files)
# dataset = dataset.map()

if __name__ == "__main__":
    # 测试
    print("total files:", len(files))
    print("train files:", len(train_files))
    print("valid files:", len(val_files))
    print("test files:", len(test_files))

    for i, ((x, y), _) in enumerate(iter(dataset)):
        print(x.shape, y.shape)
        if i == 10:
            break
