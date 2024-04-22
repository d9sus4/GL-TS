import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


class LNT:
    def __init__(self, feat_idx=None, clf=False):
        self.model = LinearRegression()
        self.theta = None
        self.feat_idx = feat_idx
        self.clf = clf

    def fit(self, X, y):
        if self.feat_idx:
            X = X[:, self.feat_idx]
        X_centered = X - np.mean(X, axis=0)
        if self.clf:
            y_binary = self.split_label(y)
            y_centered = y_binary - np.mean(y_binary, axis=0)
        else:
            y_centered = y - np.mean(y)

        self.model.fit(X_centered, y_centered)
        self.theta = self.model.coef_

    def transform(self, X):
        if self.feat_idx:
            X = X[:, self.feat_idx]
        res = X - np.mean(X, axis=0)
        new_feature = res @ self.theta
        new_feature = new_feature[:, np.newaxis]
        return new_feature

    # split labels into 2 categories
    def split_label(self, y):
        def helper(labels, group1, group2):
            if len(labels) == 0:
                if len(group1) == 0 or len(group2) == 0:
                    pass
                elif all(group1 != prev_group2 for prev_group2 in record):
                    record.append(group2)
                    res[f'{str(group1)}_{str(group2)}'] = np.vectorize(lambda x: 1 if x in group1 else 0)(y)

            else:
                label = labels[0]
                helper(labels[1:], group1 + [label], group2)
                helper(labels[1:], group1, group2 + [label])

        labels = np.unique(y)
        record = []
        res = pd.DataFrame()

        helper(labels, [], [])
        res = res.to_numpy().astype(np.float64)
        res = np.squeeze(res)
        # print(res.shape)
        return res


if __name__ == "__main__":
    X_train = np.random.rand(100, 1000)
    X_test = np.random.rand(80, 1000)
    y_train = np.random.rand(100) > 0.5
    y_test = np.random.rand(80) > 0.5

    feat_dim_combination = [[0, 5, 10, 15, 20],
                            [1, 6, 11, 16, 21]]
    X_train_lnt = []
    X_test_lnt = []

    for com in feat_dim_combination:
        lnt = LNT(feat_idx=com, clf=True)
        lnt.fit(X_train, y_train)
        X_train_lnt.append(lnt.transform(X_train))
        X_test_lnt.append(lnt.transform(X_test))

    X_train_lnt = np.hstack(X_train_lnt)
    X_train = np.hstack((X_train, X_train_lnt))

    X_test_lnt = np.hstack(X_test_lnt)
    X_test = np.hstack((X_test, X_test_lnt))  # Corrected line
