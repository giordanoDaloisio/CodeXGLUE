# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import sys, json
import numpy as np


def read_answers(filename):
    answers = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            answers[js["url"]] = js["idx"]
    return answers


def read_predictions(filename):
    predictions = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            predictions[js["url"]] = js["answers"]
    return predictions


def calculate_mrr1(answers, predictions):
    scores = []
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for url {}.".format(key))
            sys.exit()
        if predictions[key][0] == answers[key]:
            scores.append(1)
        else:
            scores.append(0)
    result = {}
    result["MRR@1"] = round(np.mean(scores), 4)
    return result


def calculate_mrr5(answers, predictions):
    scores = []
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for url {}.".format(key))
            sys.exit()
        for rank, idx in enumerate(predictions[key]):
            if idx == answers[key]:
                if rank < 5:
                    scores.append(1 / (rank + 1))
                else:
                    scores.append(0)
                break
            elif rank == len(predictions[key]) - 1:
                scores.append(0)
    result = {}
    result["MRR@5"] = round(np.mean(scores), 4)
    return result


def calculate_scores(answers, predictions):
    scores = []
    # scores_1 = []
    # scores_5 = []
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for url {}.".format(key))
            sys.exit()
        flag = False
        for rank, idx in enumerate(predictions[key]):
            if idx == answers[key]:
                scores.append(1 / (rank + 1))
                flag = True
                # if rank < 5:
                #     scores_5.append(1/(rank+1))
                #     if rank == 0:
                #         scores_1.append(1)
                #     else:
                #         scores_1.append(0)
                # else:
                #     scores_5.append(0)

        if flag is False:
            scores.append(0)
    result = {}
    result["MRR"] = round(np.mean(scores), 4)
    # result['MRR@1'] = round(np.mean(scores_1),4)
    # result['MRR@5'] = round(np.mean(scores_5),4)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate leaderboard predictions for NL-code-search-Adv dataset."
    )
    parser.add_argument(
        "--answers", "-a", help="filename of the labels, in txt format."
    )
    parser.add_argument(
        "--predictions",
        "-p",
        help="filename of the leaderboard predictions, in txt format.",
    )

    args = parser.parse_args()
    answers = read_answers(args.answers)
    predictions = read_predictions(args.predictions)
    scores = calculate_scores(answers, predictions)
    mrr_1 = calculate_mrr1(answers, predictions)
    mrr_5 = calculate_mrr5(answers, predictions)
    print(scores, mrr_1, mrr_5)


if __name__ == "__main__":
    main()
