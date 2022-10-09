import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--original', type=str, required=True, help='Original checkpoint')
parser.add_argument('--output', type=str, required=True, help='Output checkpoint path')
FLAGS = parser.parse_args()

def main():
    checkpoint = torch.load(FLAGS.original)

    torch.save({
        'actor_state_dict': checkpoint['actor_state_dict'],
        'critic_state_dict': checkpoint['critic_state_dict'],
    }, FLAGS.output)

if __name__ == '__main__':
    main()
