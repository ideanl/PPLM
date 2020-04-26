import argparse
from tqdm import tqdm
import sys
import os
import torch
from transformers import GPT2Model, GPT2Tokenizer

def get_avg_embedding(model, tokenizer, inp):
    enc = tokenizer.encode(inp)
    outputs = torch.zeros((1, 1023)).float().to('cuda')
    for i in tqdm(range(0, len(enc), 1023)):
        encoded = torch.tensor(enc[i:i+1023]).unsqueeze(0).long().cuda()
        outputs += torch.mean(model(encoded)[0].squeeze(0), dim=1) / (len(enc) / 1023 + 1)
    return outputs

def get_ref_embedding(model, tokenizer, emb_file):
    with open('datasets/gpt2_dataset.txt', 'r') as f:
        ref = f.read()
    ref = get_avg_embedding(model, tokenizer, ref)
    torch.save(ref, emb_file)
    return ref

def main(input_file=None, ref_emb_file=None):
    if input_file is None or ref_emb_file is None:
        print("input_file and ref_emb are required", file=sys.stederr)
        return

    model = GPT2Model.from_pretrained('gpt2-large')
    model.to('cuda')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

    if not os.path.exists(ref_emb_file):
        ref_emb = get_ref_embedding(model, tokenizer, ref_emb_file)
    else:
        ref_emb = torch.load(ref_emb_file)

    with open(input_file, 'r') as f:
        gen_passages = f.read().split('\n')

    similarities = []
    for gen in gen_passages:
        gen = get_avg_embedding(model, tokenizer, gen)
        cos = nn.CosineSimilarity(dim=0)
        sim = cos(gen, ref_emb)
        similarities.append(sim.detach().cpu().numpy())
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
        required=True,
        help="Reference embedding"
    )
    args = parser.parse_args()
    main(**vars(args))
