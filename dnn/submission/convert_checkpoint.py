import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--original', type=str, required=True, help='Original checkpoint')
parser.add_argument('--output', type=str, default='submission.ckpt', help='Output checkpoint path')
FLAGS = parser.parse_args()

def main():
    checkpoint = torch.load(FLAGS.original)
    actor_checkpoint = checkpoint['actor_state_dict']
    torch.save({
        'actor_state_dict': actor_checkpoint,
    }, FLAGS.output)
