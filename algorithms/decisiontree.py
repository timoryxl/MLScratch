import numpy as np


def class_counts(rows, target_index):
    counts = dict()
    for row in rows:
        label = row[target_index]
        counts[label] = counts.get(label, 0) + 1
    return counts


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if isinstance(val, int):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = '==' if isinstance(self.value, int) else '>='
        return 'Is Column No.{column} {condition} {value}?'.format(column=self.column,
                                                                   condition=condition,
                                                                   value=self.value)


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def impurity(rows, metric):
    counts = class_counts(rows)
    value = 1
    if metric.lower() == 'gini':
        for label in counts:
            p = counts[label]/float(len(rows))
            value -= p**2
    elif metric.lower() == 'entropy':
        for label in counts:
            p = counts[label]/float(len(rows))
            value -= p**np.log2(p)
    return value
