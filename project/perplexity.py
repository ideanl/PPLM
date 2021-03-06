import os
import math
import contextlib
from tqdm import tqdm
import sys
import json
import requests
import random
from transformers.modeling_gpt2 import GPT2LMHeadModel
import run_pplm
from run_pplm import run_pplm_example
import argparse
from transformers import GPT2Tokenizer
import nltk
import torch
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def setup_eval_dataset():
    r = requests.get('https://www.gutenberg.org/cache/epub/4357/pg4357.txt', stream=True)

    if not os.path.exists('./eval'):
        os.mkdir('eval')

    if not os.path.exists('./eval/fairy_tales.txt'):
        with open('./eval/fairy_tales.txt', 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching eval dataset of fairy tales", total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)

def perplexity_setup():
    setup_eval_dataset()

    with open('./eval/fairy_tales.txt') as f:
      data = f.read()
      data = data.split('\n\n\n\n')
      data=[s.replace('\n', ' ').replace('  ', '. ') for s in data[6:-4]]

    sentences = []
    for passage in data:
      sentences += sent_tokenize(passage)

    random.shuffle(sentences)
    
    return sentences

def perplexity_loop(bow=False, **kwargs):
    if bow:
        params = {'gm_scale': 0.95, 'kl_scale': 0.01, 'stepsize': 0.03, 'num_iterations': 3}
    else:
        params = {'gm_scale': 0.90, 'kl_scale': 0.02, 'stepsize': 0.04, 'num_iterations': 20}
    params['max_num'] = 20
    params['in_dir'] = './out'

    params.update((k, v) for k, v in kwargs.items() if v is not None)

    type_m = "Bag of Words" if bow else "Discriminator"
    print(f"Running perplexities for {type_m} with parameters: ", params)

    sentences = perplexity_setup()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    max_num = min(params['max_num'], len(sentences))

    perplexities = []
    for j, sentence in enumerate(sentences[0:max_num]):
        loss = 0
        num_tokens = 0
        enc = tokenizer.encode(sentence)
        for i in range(len(enc)):
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    text = tokenizer.decode(enc[:i])
                    if bow:
                        probs = get_bow_probs(params['in_dir'], text, gm_scale=params['gm_scale'], kl_scale=params['kl_scale'], stepsize=params['stepsize'], num_iterations=params['num_iterations'])
                    else:
                        probs = get_discriminator_probs(params['in_dir'], text)
            prob_true = (probs[0][enc[i]] / torch.sum(probs)).detach().cpu().numpy()
            loss -= math.log(prob_true)
            num_tokens += 1

            # clear variables due to a memory leak in the library
            for x in dir(run_pplm):
                if not x.startswith("__"):
                    del x
        perplexities.append(math.exp(loss / num_tokens))
        print(f"Perplexity @ sentence {j+1}/{max_num}: {perplexities[-1]}")
        print(f"Mean Perplexity @ sentence {j+1}/{max_num}: {sum(perplexities) / len(perplexities)}")

    return perplexities

def get_bow_probs(bow_dir, text, gm_scale=0.95, kl_scale=0.01, stepsize=0.03, num_iterations=3):
    return run_pplm_example(
        cond_text=text,
        num_samples=1,
        bag_of_words=f"{bow_dir}/story_bow.txt",
        length=1,
        stepsize=stepsize,
        sample=True,
        uncond=(True if text == '' else False),
        num_iterations=num_iterations,
        window_length=5,
        gamma=1.5,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        verbosity='quiet'
    )[1]

def get_discriminator_probs(discrim_dir, text, gm_scale=0.90, kl_scale=0.02, stepsize=0.04, num_iterations=20):
    return run_pplm_example(
        cond_text=text,
        num_samples=1,
        discrim='generic',
        discrim_meta=f"{discrim_dir}/generic_classifier_head_meta.json",
        discrim_weights=f"{discrim_dir}/generic_classifier_head_epoch_10.pt",
        class_label='1',
        length=1,
        stepsize=stepsize,
        sample=True,
        uncond=(True if i == 0 else False),
        num_iterations=num_iterations,
        gamma=1,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        verbosity='quiet'
    )[1]

def main(bow=False, discriminator=False, perplexity=True, **kwargs):
    if (bow and discriminator) or (not bow and not discriminator):
        print("Must specify exactly one of BoW or discriminator", file=sys.stderr)
        return

    if perplexity:
        perplexities = perplexity_loop(bow=bow, **kwargs)
        out_file = kwargs.get('out_file', './eval/perplexity.json')
        with open(out_file, 'w') as f:
            print(perplexities)
            json.dump(perplexities, f)
        print(f"Perplexities outputted to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bow",
        type=bool,
        default=False,
        help="Use BoW model"
    )
    parser.add_argument(
        "--discriminator",
        type=bool,
        default=False,
        help="Use discriminator model"
    )
    parser.add_argument(
        "--perplexity",
        type=bool,
        default=True,
        help="Use discriminator model"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="./eval/perplexity.json",
        help="In directory of necessary model files"
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        default="./out",
        help="In directory of necessary model files"
    )
    parser.add_argument(
        "--max_num",
        type=int,
        default=20,
        help="Nax number of examples to evaluate for"
    )
    parser.add_argument(
        "--gm_scale",
        type=int,
        default=None,
        help="GM Scale"
    )
    parser.add_argument(
        "--kl_scale",
        type=int,
        default=None,
        help="KL Scale"
    )
    parser.add_argument(
        "--stepsize",
        type=int,
        default=None,
        help="Stepsize"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=None,
        help="Number of iterations"
    )
    args = parser.parse_args()
    main(**vars(args))

