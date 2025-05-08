# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import sys
import json
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
import pandas as pd
import os


def read_answers(filename):
    answers = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            answers[js["idx"]] = js["target"]
    return answers


def read_predictions(filename):
    predictions = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            idx, label = line.split()
            predictions[int(idx)] = int(label)
    return predictions


def calculate_scores(answers, predictions):
    Acc = []
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        Acc.append(answers[key] == predictions[key])

    scores = {}
    scores["Acc"] = np.mean(Acc)
    scores["f1"] = f1_score(list(answers.values()), list(predictions.values()))
    scores["mcc"] = matthews_corrcoef(
        list(answers.values()), list(predictions.values())
    )
    return scores


def main():
    # import argparse

    # parser = argparse.ArgumentParser(
    #     description="Evaluate leaderboard predictions for Defect Detection dataset."
    # )
    # parser.add_argument(
    #     "--answers", "-a", help="filename of the labels, in txt format."
    # )
    # parser.add_argument(
    #     "--predictions",
    #     "-p",
    #     help="filename of the leaderboard predictions, in txt format.",
    # )

    # args = parser.parse_args()

    answers = read_answers('../dataset/test.jsonl')
    df = pd.read_csv('../../../analysis/experiment_values.csv')
    for logs in os.listdir('../code/saved_models_graph/'):
        if os.path.isdir(os.path.join('../code/saved_models_graph/', logs)) or 'cuda' in logs:
            continue
        task = 'Defect Prediction Graph'
        if 'prune4' in logs:
            compression = 'Pruning 0.4'
        elif 'prune6' in logs:
            compression = 'Pruning 0.6'
        elif 'prune' in logs:
            compression = 'Pruning 0.2'
        elif 'quanf8' in logs:
            compression = 'Quantization (quanto-qfloat8)'
        elif 'quant4' in logs:
            compression = 'Quantization (quanto-qint4)'
        elif 'quant' in logs:
            compression = 'Quantization (quanto-qint8)'
        else:
            compression = 'No One'
        predictions = read_predictions(os.path.join('../code/saved_models_graph/', logs))
        scores = calculate_scores(answers, predictions)
        df = pd.concat([df, pd.DataFrame({
            'Task': [task, task, task], 
            'Compression Method': [compression, compression, compression], 
            'Parameter': ["Accuracy", "F1", "MCC"], 
            'Value': [scores['Acc'], scores['f1'], scores['mcc']]
        })], ignore_index=True)  
        df.to_csv('../../../analysis/experiment_values.csv', index=False)
        print(scores)


if __name__ == "__main__":
    main()
