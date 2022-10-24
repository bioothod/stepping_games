import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--original', type=str, required=True, help='Original checkpoint')
parser.add_argument('--want_critic', action='store_true', help='Whether to store critic weights')
parser.add_argument('--output', type=str, required=True, help='Output checkpoint path')
FLAGS = parser.parse_args()

def main():
    checkpoint = torch.load(FLAGS.original)

    output_dict = {
        'actor_state_dict': checkpoint['actor_state_dict'],
    }
    if FLAGS.want_critic:
        output_dict['critic_state_dict'] = checkpoint['critic_state_dict']

    torch.save(output_dict, FLAGS.output)
    print(f'saved to {FLAGS.output}')

if __name__ == '__main__':
    main()
