import os
import gpt_2_simple as gpt2
import requests
import argparse

def main(steps=200):
    model_name="355M"
    if not os.path.isdir(os.path.join('models', model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/

    file_name = "./datasets/DatasetV3.txt"
        
    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
                  file_name,
                  model_name=model_name,
                  steps=steps)   # steps is max number of training steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of iterations"
    )
    args = parser.parse_args()
    main(**vars(args))
