import random
import re
from collections import defaultdict
from pathlib import Path

DATASET_DIR = Path("./dataset")


def main(dataset_name):
    save_dir = DATASET_DIR / dataset_name
    save_dir.mkdir(exist_ok=True, parents=True)

    word_def = defaultdict(lambda: [])

    modes = ["train", "valid", "test"]
    for mode in modes:
        with (DATASET_DIR / "ishiwatari" / dataset_name / f"{mode}.txt").open() as f:
            for line in f:
                word, _, _, definition, *_ = line.strip().split("\t")
                word = word.rsplit("%", 1)[0].lstrip().rstrip()
                definition = (
                    definition.replace(" .", ".")
                    .replace(" ,", ",")
                    .replace(" ;", ";")
                    .replace("( ", "(")
                    .replace(" )", ")")
                    .replace(" '", "'")
                )
                definition = re.sub(
                    r"`` (.*?)''", lambda x: x.group(1).capitalize(), definition
                )
                definition = re.sub(r"‘\s*(.*?)\s*’", r"’\1’", definition)
                definition = definition.lstrip().rstrip()
                word_def[word].append(definition)

    all_words = sorted(word_def.keys())

    def process(filename, words):
        num = 0
        lines = []
        for word in words:
            definitions = word_def[word]
            num += len(definitions)
            lines += [f"{word}\t{definition}" for definition in definitions]

        (save_dir / filename).write_text("\n".join(lines))
        return num

    print("sum of\tall lines:\t", process("all.tsv", all_words))

    random.shuffle(all_words)
    train_words = all_words[: len(all_words) * 8 // 10]
    valid_words = all_words[len(all_words) * 8 // 10 : len(all_words) * 9 // 10]
    test_words = all_words[len(all_words) * 9 // 10 :]

    print("sum of\twords:\t", len(all_words))
    print("sum of\ttrain words:\t", len(train_words))
    print("sum of\tvalid words:\t", len(valid_words))
    print("sum of\ttest words:\t", len(test_words))

    print("sum of\ttrain lines:\t", process("train.tsv", train_words))
    print("sum of\tvalid lines:\t", process("valid.tsv", valid_words))
    print("sum of\ttest lines:\t", process("test.tsv", test_words))


if __name__ == "__main__":
    main("oxford")
    # main("wiki")
    # main("slang")
