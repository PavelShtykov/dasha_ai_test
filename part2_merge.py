import json
import glob
from typing import List, Tuple
import argparse
from collections import defaultdict


def union_sentence(text: str, spans: List[Tuple[int, int, str]]) -> dict:
    entities = [
        {
            "start": s[0],
            "end": s[1],
            "value": text[s[0]:s[1]],
            "entity": s[2]
        }
        for s in spans if s[2] is not None
    ]

    # union neighbor spans
    i = 0
    while True:
        if i == len(entities) - 1 or len(entities) == 0:
            break

        if entities[i]["end"] + 1 == entities[i+1]["start"] \
                and entities[i]["entity"] == entities[i+1]["entity"]:

            entities[i+1] = {
                "start": entities[i]["start"],
                "end": entities[i+1]["end"],
                "value": entities[i]["value"] + " " + entities[i+1]["value"],
                "entity": entities[i]["entity"]
            }
            entities.pop(i)
        else:
            i += 1

    return {
        "text": text,
        "entities": entities
    }


def atis_sentence(dataset: List[dict], mapping: dict) -> Tuple[str, List[Tuple[int, int, str]]]:
    res = []
     
    for sent in dataset:
        arg = (
            sent["text"],
            [
                (
                    0 if e["start"] == 1 else e["start"],
                    e["end"] - 1 if e["start"] == 1 else e["end"],
                    mapping[e["entity"]]
                )
                for e in sent["entities"]
            ]
        )
        res.append(union_sentence(*arg))

    return res


def cross_intersection(ents: List[Tuple[int, int, str]]) -> bool:
    def intrs(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return not (b[1] < a[0] or a[1] < b[0])

    bounds = []
    for ent in ents:
        bound = (ent[0], ent[1])
        if any(map(lambda x: intrs(bound, x), bounds)):
            return True
        bounds.append(bound)

    return False


def rest_sentence(dataset: List[dict], mapping: dict) -> Tuple[str, List[Tuple[int, int, str]]]:
    res = []

    for sent in dataset:
        arg = ( 
            sent["userInput"]["text"],
            [
                (
                    l["valueSpan"].get("startIndex", 0),
                    l["valueSpan"]["endIndex"],
                    mapping[l["slot"]]
                )
                for l in sent.get("labels", [])
            ]
        )
        if cross_intersection(arg[1]):
            continue

        res.append(union_sentence(*arg))

    return res
    

def union_dataset_stat(test_path: str, train_path: str) -> dict:
    stat = defaultdict(set)

    for path in [test_path, train_path]:
        with open(path, "r") as f:
            dataset = json.load(f)
        
        for sent in dataset:
            for ent in sent["entities"]:
                stat[ent["entity"]].add(ent["value"])

    stat = {k: sorted(list(v)) for k, v in stat.items()}

    return stat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapper",
        help="path to mapper json",
        type=str,
        default="res/part_2/entities_map.json"
    )
    args = parser.parse_args()

    with open(args.mapper, "r") as f:
        mapper = json.load(f)

    union_test, union_train = [], []

    # -----ATIS-----
    with open("data/atis/test.json", "r") as f:
        atis_test = json.load(f)

    atis_test = atis_test["rasa_nlu_data"]["common_examples"]
    union_test += atis_sentence(atis_test, mapper["atis_entities_map"])

    with open("data/atis/train.json", "r") as f:
        atis_train = json.load(f)

    atis_train = atis_train["rasa_nlu_data"]["common_examples"]
    union_train += atis_sentence(atis_train, mapper["atis_entities_map"])

    # ---------REST----------
    with open("data/restaurant8k/test.json", "r") as f:
        rest_test = json.load(f)

    union_test += rest_sentence(rest_test, mapper["rest_entities_map"])

    for path in glob.glob("data/restaurant8k/train_*.json"):
        with open(path, "r") as f:
            rest_train = json.load(f)

        union_train += rest_sentence(rest_train, mapper["rest_entities_map"])

    # dump merged dataset
    with open("res/part_2/union_test.json", "w") as f:
        json.dump(union_test, f, indent=4)

    with open("res/part_2/union_train.json", "w") as f:
        json.dump(union_train, f, indent=4)

    # count stat on union dataset
    with open("res/part_2/union_stat.json", "w") as f:
        stat = union_dataset_stat("res/part_2/union_test.json",
            "res/part_2/union_train.json")
        json.dump(stat, f, indent=4)