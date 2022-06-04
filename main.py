from preprocessing import make_train_files, make_test_files
from bert_model import BertModel
from transformers import TFBertModel
from constants import REBUILD, REMAKE, TESTED, SCORE
import shlex
import subprocess


def official_f1():
    with open(SCORE + 'result.txt', 'r', encoding='utf-8') as f:
        macro_result = list(f)[-1]
        macro_result = macro_result.split(":")[1].replace(">>>", "").strip()
        macro_result = macro_result.split("=")[1].strip().replace("%", "")
        macro_result = float(macro_result) / 100
    return macro_result


def main():
    if REBUILD == 1:
        make_train_files()
        make_test_files()
    encoder = TFBertModel.from_pretrained("bert-base-uncased")
    model = BertModel(encoder)
    model.build()
    model.predict()


def print_f1():
    print("macro-averaged F1 = {}%".format(official_f1() * 100))


if __name__ == '__main__':
    if REMAKE == 1:
        main()
    if TESTED == 1:
        print_f1()
