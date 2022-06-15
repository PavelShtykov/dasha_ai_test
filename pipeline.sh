BASE_ATIS=data/atis
BASE_REST=data/restaurant8k
RES_DIR=res
MAPPER=utils/entities_map.json

if ! test -d "$BASE_ATIS"; then
    echo "$BASE_ATIS don't exist"
    exit
fi

if ! test -d "$BASE_REST"; then
    echo "$BASE_REST don't exist"
    exit
fi

rm -r "$RES_DIR"
mkdir "$RES_DIR"
echo "make $RES_DIR directory"

mkdir "$RES_DIR/part1"
python3 "part1_analysis.py" "$BASE_ATIS" "$BASE_REST" "$RES_DIR/part1"
printf "\n OK: part1_analysis.py \n"

if ! test -f "$MAPPER"; then
    printf "$MAPPER don't exist"
    exit
fi

mkdir "$RES_DIR/part2"
python3 "part2_merge.py" "$BASE_ATIS" "$BASE_REST" "$RES_DIR/part2" "$MAPPER"
printf "\n OK: part2_merge.py \n"


# mkdir -p "$RES_DIR/part3/flair"
# python3 "part3_train_flair.py" "$RES_DIR/part2/union_test.json" \
# "$RES_DIR/part2/union_train.json" "$RES_DIR/part2/union_dev.json" "$RES_DIR/part3/flair"

# printf "\n OK: part3_train_flair.py \n"

# mkdir -p "$RES_DIR/part3/flair"
python3 "part3_train_w2ner.py" "$RES_DIR/part2/union_test.json" \
"$RES_DIR/part2/union_train.json" "$RES_DIR/part2/union_dev.json" "w2ner/data/example"

printf "\n OK: preprocessing data for w2ner \n"

cd "w2ner"
python3 "main.py" --config "config/example.json"