from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import json
import spacy
import os
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from sympy import arg


def create_w2ner_dataset(test_path: str, train_path: str, dev_path: str, save_dir: str) -> Corpus:
    """
    Transform union-type dataset to w2ner-type dataset
    """
    with open(test_path, "r") as f:
        old_test = json.load(f)
    
    with open(train_path, "r") as f:
        old_train = json.load(f)

    with open(dev_path, "r") as f:
        old_dev = json.load(f)

    nlp = spacy.blank('en')

    grid = [("test.json", old_test),
            ("train.json", old_train),
            ("dev.json", old_dev)]
    for file_name, old_dataset in grid:
        with open(os.path.join(save_dir, file_name), "w") as f:
            new_dataset = []
            for sent in old_dataset:
                doc = nlp(sent["text"])

                try:
                    doc.set_ents([
                        doc.char_span(e["start"], e["end"], e["entity"])
                        for e in sent["entities"]
                    ])
                except:
                    continue
                
                res = {"sentence": [], "ner": []}
                for idx, tok in enumerate(doc):
                    res["sentence"].append(str(tok))
                    label = tok.ent_iob_
                    if tok.ent_iob_ == "B":
                        res["ner"].append({"index": [idx], "type": tok.ent_type_})
                    if tok.ent_iob_ == "I":
                        res["ner"][-1]["index"].append(idx)
                new_dataset.append(res)

            json.dump(new_dataset, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "union_test",
        help="path to union test",
        type=str,
    )
    parser.add_argument(
        "union_train",
        help="path to union train",
        type=str,
    )
    parser.add_argument(
        "union_dev",
        help="path to union dev",
        type=str,
    )
    parser.add_argument(
        "save_dir",
        help="path to save to save logs",
        type=str,
    )
    args = parser.parse_args()

    corpus = create_w2ner_dataset(
        test_path=args.union_test,
        train_path=args.union_train,
        dev_path=args.union_dev,
        save_dir=args.save_dir
    )

