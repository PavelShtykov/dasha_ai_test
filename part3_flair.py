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


def create_flair_dataset(test_path: str, train_path: str, save_dir: str) -> Corpus:
    with open(test_path, "r") as f:
        old_test = json.load(f)
    
    with open(train_path, "r") as f:
        old_train = json.load(f)

    nlp = spacy.blank('en')

    grid = [("test.txt", old_test),
            ("train.txt", old_train)]
    for file_name, old_dataset in grid:
        with open(os.path.join(save_dir, file_name), "w") as f:
            for sent in old_dataset:
                doc = nlp(sent["text"])

                try:
                    doc.set_ents([
                        doc.char_span(e["start"], e["end"], e["entity"])
                        for e in sent["entities"]
                    ])
                except:
                    continue
                
                for tok in doc:
                    if "\\" in str(tok):
                        continue
                    label = tok.ent_iob_
                    if tok.ent_iob_ != "O":
                        label += '-' + tok.ent_type_
                    f.write(f"{tok} {label}\n")
                f.write("\n")

    columns = {0: 'text', 1: 'ner'}
    corpus: Corpus = ColumnCorpus(save_dir, columns,
                                    train_file='train.txt',
                                    test_file='test.txt'
    )

    return corpus


def draw_plots(log: dict, dir_to_save: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(log["train_loss_history"], label="train")
    axes[0].plot(log["dev_loss_history"], label="val")
    axes[0].grid()
    axes[0].legend()
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Epoch Loss")

    axes[1].plot(log["dev_score_history"], label="val")
    axes[1].grid()
    axes[1].legend()
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("F1-score")
    axes[1].set_title("Epoch Metric")

    plt.suptitle("Flair model learning curves")
    plt.savefig(os.path.join(dir_to_save, "flair_curves.svg"))


if __name__ == "__main__":
    save_dir = "res/part_3/flair"
    corpus = create_flair_dataset(
        test_path="res/part_2/union_test.json",
        train_path="res/part_2/union_train.json",
        save_dir=save_dir
    )

    print("\nExample of tagged string from test split:")
    print(corpus.test[10].to_tagged_string('ner'))
    print(corpus.test[100].to_tagged_string('ner'))
    print(corpus.test[1000].to_tagged_string('ner'))

    label_type = "ner"
    
    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type)
    print(label_dict, "\n")

    # 4. initialize embedding stack with Flair and GloVe
    embedding_types = [
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            use_crf=True)

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 7. start training
    log = trainer.train('res/part_3/flair/model',
                learning_rate=0.1,
                mini_batch_size=32,
                max_epochs=15,
                num_workers=2)

    print("Example of prediction:")
    print(corpus.test[100])
    print(corpus.test[1000])
    print(corpus.test[500])
    print(corpus.test[1200])
    print(corpus.test[600])
    print(corpus.test[200])
    print(corpus.test[1300])
    print(corpus.test[650])

    draw_plots(log, save_dir)


