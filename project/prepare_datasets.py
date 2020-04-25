import numpy as np
import os
import sys
import requests
from tqdm import tqdm
import json
import argparse
import re
import nltk
from run_pplm_discrim_train import train_discriminator
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from transformers import GPT2Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt')

def _load_split(data_dir, source, split, n=np.inf):
    path = os.path.join(data_dir, f'{source}.{split}.jsonl')
    texts = []
    for i, line in enumerate(open(path)):
        if i >= n:
            break
        texts.append(json.loads(line)['text'])
    return texts

def prepare(stories_path):
    with open(stories_path) as f:
      text = f.read()
      stories = text.split('=========\n')

    """### Get general dataset of non-children's stories"""
    subdir = 'webtext_data'
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\','/') # needed for Windows

    for ds in [
        'webtext',
    ]:
        for split in ['train', 'valid', 'test']:
            filename = ds + "." + split + '.jsonl'
            if os.path.exists(os.path.join(subdir, filename)):
                continue

            r = requests.get("https://storage.googleapis.com/gpt-2/output-dataset/v1/" + filename, stream=True)

            with open(os.path.join(subdir, filename), 'wb') as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)
    train_texts = _load_split(subdir, 'webtext', 'train')
    valid_texts = _load_split(subdir, 'webtext', 'valid')

    return stories, (train_texts, valid_texts)

def write_to_discriminator(tokenizer, fp, text, label, cutoff=True, stride=0, max_length=99):
    c = 0

    encoding = tokenizer.encode(tokenizer.decode(tokenizer.encode(text.replace("\"", "").replace('\t', '    '))))

    if cutoff:
        c = 1
        fp.write(str(label) + '\t' + ''.join(tokenizer.decode(encoding[0:max_length])) + '\n')
    else:
        i = 0
        while i < len(encoding):
            c += 1
            start = max(i-stride, 0)
            end = start + max_length
            if end > len(encoding):
                end = len(encoding)
            fp.write(str(label) + '\t' + ''.join(tokenizer.decode(encoding[start:end])) + '\n')
            i = end
    return c


def setup_discriminator(out_dir, stories, train_texts):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    with open(f"{out_dir}/discriminator.txt", 'w') as d:
        pos_lines = 0
        for passage in tqdm(stories):
          passage = passage.strip().replace('\n', '. ').strip()
          sentences = sent_tokenize(passage)
          for i in range(0, len(sentences), 1):
            if sentences[i] == '.':
              continue
            pos_lines += write_to_discriminator(tokenizer, d, ' '.join(sentences[i:i+1]), 1)

        neg_lines = 0
        for i, text in enumerate(tqdm(train_texts[0:(2000)])):
          text = text.strip().replace('\n', '. ').strip()
          sentences = sent_tokenize(text)

          for i in range(0, len(sentences), 1):
            if sentences[i] == '.':
              continue
            neg_lines += write_to_discriminator(tokenizer, d, ' '.join(sentences[i:i+1]), 0)

def train_discrim(out_dir, stories, train_texts):
    setup_discriminator(out_dir, stories, train_texts)
    train_discriminator(dataset='generic', dataset_fp="{out_dir}/discriminator.txt", save_model=True, output_fp=out_dir)


"""# PPLM Bag of Words
This bag of words is formed by using a vectorizer that excludes words with a high document frequency, where most of the documents (by far) are from webtext. The first document is fairy tales from Project Gutenberg. The descriptive words in this document are the target for this Bag of Words list
"""
def setup_bow(out_dir, stories, train_texts, max_df=0.1, num_words=200):
    counter = CountVectorizer('content', strip_accents='unicode', lowercase=True, max_df=max_df)
    X = counter.fit_transform(['\n'.join(stories)] + train_texts)

    words_freq = [(word, float(X[0, idx])) for idx, word in enumerate(tqdm(counter.get_feature_names()))]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    words, freqs = zip(*words_freq[0:num_words])

    with open(f"{out_dir}/story_bow.txt", 'w') as f:
      for word in words:
          f.write(word + '\n')


def main(
        stories_path='./datasets/DatasetV3.txt',
        out_dir='./out',
        discriminator=False,
        bow=False,
        max_df=0.1,
        num_words=200
):
    if stories_path is None:
        print("stories_path argument missing")
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    stories, (train_texts, _) = prepare(stories_path)
    if bow:
        setup_bow(out_dir, stories, train_texts, max_df=max_df, num_words=num_words)
    if discriminator:
        train_discrim(out_dir, stories, train_texts)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--stories_path",
    type=str,
    default="./datasets/DatasetV3.txt",
    help="Dataset of stories path"
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="./out",
    help="Output dir of BoW or discriminator saved files"
)
parser.add_argument(
    "--bow",
    type=bool,
    default=False,
    help="Generate BoW wordlist"
)
parser.add_argument(
    "--discriminator",
    type=bool,
    default=False,
    help="Train discriminator"
)
parser.add_argument(
    "--max_df",
    type=int,
    default=0.1,
    help="Max DF for the BoW vectorizer"
)
parser.add_argument(
    "--num_words",
    type=int,
    default=200,
    help="Number of words for the BoW list"
)
args = parser.parse_args()
main(**vars(args))
