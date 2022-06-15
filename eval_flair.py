from flair.models import SequenceTagger
from flair.data import Sentence
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "weights",
        help="path to pretrained weights of Tagger",
        type=str,
    )
    args = parser.parse_args()

    # load pretrained tagger
    tagger = SequenceTagger.load(args.weights)
    
    # create Sentence for flair model
    sentence = Sentence('George Washington went to restaurant at 12:04 on friday')

    # predict
    tagger.predict(sentence)

    # print result
    print(sentence.to_tagged_string())