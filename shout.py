import argparse

def main(args):
    conf = load_config("config.json")
    audio = load_audio(conf['sample_directory'], conf['formats'])

    model = Gan()

def parse_args():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--mode', default='train', help='train/gen')

    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()

    from audio_model import *
    from frequency_filters import *
    from note_maps import *
    from utils import *

    main(args)
