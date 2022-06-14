import json
import glob
from collections import defaultdict
from typing import Tuple


def intersection(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (b[1] < a[0] or a[1] < b[0])


if __name__ == "__main__":
    # --------ATIS--------
    atis_stat = {
        "nested": False,
        "intents": set(),
        "entities_examples": defaultdict(set),
        "entities_map": {}
    }

    for path in glob.glob("./data/atis/*.json"):
        with open(path, "r") as f:
            data = json.load(f)["rasa_nlu_data"]["common_examples"]

        for sent in data:
            curr_intents = set(sent["intent"].split("+"))
            atis_stat["intents"].update(curr_intents)

            curr_entities = {
                d["entity"]: {d["value"].strip()} | atis_stat["entities_examples"][d["entity"]]
                for d in sent["entities"]
            }
            atis_stat["entities_examples"].update(curr_entities)

            curr_bounds = set()
            for v in sent["entities"]:
                bound = (
                    0 if v["start"] == 1 else v["start"],
                    v["end"] - 1 if v["start"] == 1 else v["end"]
                )
                if any(map(lambda b: intersection(bound, b), curr_bounds)):
                    atis_stat["nested"] = True
                    print(sent)
                curr_bounds.add(bound)

    # cast sets to list, sort and dump
    atis_stat["entities_map"] = {
        k: None
        for k in sorted(atis_stat["entities_examples"])
    }
    atis_stat["intents"] = sorted(list(atis_stat["intents"]))
    atis_stat["entities_examples"] = {
        k: list(v)[:10]
        for k, v in sorted(atis_stat["entities_examples"].items())
    }

    with open("res/part_1/atis.json", "w") as f:
        json.dump(atis_stat, f, indent=4)

    # --------RESTAURANT----------
    rest_stat = {
        "nested": False,
        "intents": set(),
        "entities_examples": defaultdict(set),
        "entities_map": {}
    }

    for path in glob.glob("./data/restaurant8k/*"):
        with open(path, "r") as f:
            data = json.load(f)

        for sent in data:
            if "context" in sent.keys() and len(sent["context"]) > 0:
                curr_context = set(sent["context"]["requestedSlots"])
                rest_stat["intents"].update(curr_context)

            if "labels" in sent.keys():
                curr_entities = {}
                curr_bounds = list()
                for v in sent["labels"]:
                    bound = (
                        v["valueSpan"].get("startIndex", 0),
                        v["valueSpan"]["endIndex"]
                    )
                    curr_entities[v["slot"]] = {sent["userInput"]["text"][bound[0]:bound[1]].strip()} | \
                        rest_stat["entities_examples"][v["slot"]]

                    if any(map(lambda b: intersection(bound, b), curr_bounds)):
                        rest_stat["nested"] = True
                        print(sent)
                    curr_bounds.append(bound)

                rest_stat["entities_examples"].update(curr_entities)

    # cast sets to list, sort and dump
    rest_stat["entities_map"] = {
        k: None
        for k in sorted(rest_stat["entities_examples"])
    }
    rest_stat["intents"] = sorted(list(rest_stat["intents"]))
    rest_stat["entities_examples"] = {
        k: list(v)[:30]
        for k, v in sorted(rest_stat["entities_examples"].items())
    }

    with open("res/part_1/rest.json", "w") as f:
        json.dump(rest_stat, f, indent=4)
