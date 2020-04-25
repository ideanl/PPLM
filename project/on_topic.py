import argparse
import sys
import os
import torch
from transformers import GPT2Model, GPT2Tokenizer

def get_avg_embedding(inp):
    enc = tokenizer.encode(inp, add_special_tokens=True)
    outputs = torch.zeros((1, 1023)
    for i in range(0, len(enc), 1023):
        encoded = torch.Tensor(outputs[i:i+1023])[0]
        outputs += model(encoded) / (len(encoding) / 1023 + 1)
    return outputs

def get_ref_embedding(emb_file):
    with open('datasets/gpt2_dataset.txt') as f:
        ref = f.read()
    ref = get_avg_embedding(ref)
    torch.save(ref, emb_file)
    return ref

def main(input_file=None, ref_emb_file=None):
    if input_file is None or ref_emb_file is None:
        print("input_file and ref_emb are required", file=sys.stederr)
        return

    if not os.path.exists(ref_emb_file):
        ref_emb = get_ref_embedding(ref_emb_file)
    else:
        ref_emb = torch.load(ref_emb_file)

    with open(input_file, 'r') as f:
        gen_passages = f.read().split('\n')

    model = GPT2Model.from_pretrained('gpt2-large')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

    similarities = []
    for gen in gen_passages:
        gen = get_avg_embedding(gen)
        cos = nn.CosineSimilarity(dim=0)
        sim = cos(gen, ref_emb)
        similarities.append(sim)
        print(f"Similarity at {gen}/{gen_passages}: ", sim)

    print("========")
    print(similarities)


if __name__ == '__main__':
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input file of sequences"
    )
    parser.add_argument(
        "--ref_emb_file",
        required=True
        help="Reference embedding"
    )
    args = parser.parse_args()
    main(**vars(args))
