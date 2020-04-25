import gpt_2_simple as gpt2
import contextlib
import os
import argparse
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import GPT2Tokenizer
import run_pplm
from run_pplm import run_pplm_example

def gen_bow(bow_dir, text, seed=0, length=20, gm_scale=0.95, kl_scale=0.01, stepsize=0.03, num_iterations=3):
    return run_pplm_example(
        cond_text=text,
        num_samples=1,
        bag_of_words=f"{bow_dir}/story_bow.txt",
        length=length,
        stepsize=stepsize,
        sample=True,
        seed=seed,
        num_iterations=num_iterations,
        window_length=5,
        gamma=1.5,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        verbosity='quiet'
    )[0]

def gen_discrim(discrim_dir, text, seed=0, length=20, gm_scale=0.90, kl_scale=0.02, stepsize=0.04, num_iterations=20):
    return run_pplm_example(
        cond_text=text,
        num_samples=1,
        seed=seed,
        discrim='generic',
        discrim_meta=f"{discrim_dir}/generic_classifier_head_meta.json",
        discrim_weights=f"{discrim_dir}/generic_classifier_head_epoch_10.pt",
        class_label='1',
        length=length,
        stepsize=stepsize,
        sample=True,
        num_iterations=num_iterations,
        gamma=1,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        verbosity='quiet'
    )[0]

def gen_finetuned(_, text, **params):
    sess = params['sess']
    model_name = params['model_name']
    return gpt2.generate(sess, model_name=model_name, prefix=text, truncate="<|endoftext|>", return_as_list=True, length=params.get('length', 20))[0]

def gen_processor(gen_fn, in_dir, context, **params):
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            result = gen_fn(in_dir, context, **params)
            
    for x in dir(run_pplm):
        if not x.startswith("__"):
            del x

    return result

def main(bow=False, discriminator=False, finetuned=False, in_dir='./out', **kwargs):
    if bow + discriminator + finetuned != 1:
        print("Must specify exactly one of BoW or discriminator", file=sys.stderr)
        return

    if bow:
        params = {'gm_scale': 0.95, 'kl_scale': 0.01, 'stepsize': 0.03, 'num_iterations': 3}
        fn = gen_bow
    elif discriminator:
        params = {'gm_scale': 0.90, 'kl_scale': 0.02, 'stepsize': 0.04, 'num_iterations': 20}
        fn = gen_discrim
    else:
        sess = gpt2.start_tf_sess()
        params = {'sess': sess, 'model_name': '774M'}
        gpt2.load_gpt2(params['sess'])
        fn = gen_finetuned

    params['length'] = 20
    params.update((k, v) for k, v in kwargs.items() if v is not None)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

    context = ""
    for prefix in ["Once upon a time", "Every day", "But, one day", "Because of that", "Until, finally", "And, ever since then"]:
        context += prefix
        encoded = tokenizer.encode(context)
        if len(encoded) > 1024:
            context = tokenizer.decode(encoded[-1024:])

        while True:
            result = gen_processor(fn, in_dir, context, **params)
            if not finetuned:
                result = tokenizer.decode(tokenizer.encode(result)[1:]) # remove end token at beginning
            sentences = sent_tokenize(result)
            one_new_sentence = len(sent_tokenize(result[len(context):])) == 1
            is_last_end = len(sent_tokenize(sentences[-1] + ' end test')) > 1

            if one_new_sentence and not is_last_end:
                # incomplete sentence generated, need to gen longer sequences...
                params['length'] += 10
                continue
            break

        if len(sentences) > 1:
            context = ' '.join(sentences[:-1]) + ' '
        else:
            context = ""

        if is_last_end:
            context += sentences[-1] + ' '

        #print("Result: ", result)
        #print("Sentences: ", sentences)
        print("Story so far: ", context)
        print("\n")
    return context


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
        "--finetuned",
        type=bool,
        default=False,
        help="Use finetuned GPT-2"
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        default="./out",
        help="In directory of necessary model files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random number generator seed"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=20,
        help="Max sequence length"
    )
    parser.add_argument(
        "--gm_scale",
        type=float,
        default=None,
        help="GM Scale"
    )
    parser.add_argument(
        "--kl_scale",
        type=float,
        default=None,
        help="KL Scale"
    )
    parser.add_argument(
        "--stepsize",
        type=float,
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
    story = main(**vars(args))
    print("================")
    print(story)
