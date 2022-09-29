import csv
import itertools
import sys
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import RidgeCV



CHECK_POS = ["אבחנה-Nodes exam", "אבחנה-Age", "אבחנה-Positive nodes"]
TO_REMOVE = ['אבחנה-Diagnosis date', ' Hospital', 'User Name', 'אבחנה-Side', 'אבחנה-Surgery date1', 'אבחנה-Surgery date2',
             'אבחנה-Surgery date3', 'אבחנה-Surgery name1', 'אבחנה-Surgery name2','אבחנה-Surgery name3', 'אבחנה-Tumor depth'
              ,'אבחנה-Tumor width', 'surgery before or after-Activity date', 'surgery before or after-Actual activity','id-hushed_internalpatientid']
AREAS = ['ADR - Adrenals', 'BON - Bones', 'PLE - Pleura',
             'PER - Peritoneum', 'SKI - Skin', 'OTH - Other',
             'HEP - Hepatic', 'BRA - Brain', 'PUL - Pulmonary',
             'MAR - Bone Marrow', 'LYM - Lymph nodes']

sorted_dict = {"Not yet Established": 0, "Stage0": 0, "Stage0a": 0, "Stage0is": 0, "Stage1": 1, "Stage1a": 2,
               "Stage1b": 3, "Stage1c": 4, "Stage2": 5, "Stage2a": 6, "Stage3": 7, "Stage3b": 8, "Stage3c": 9,
               "Stage3a": 10, "Stage4": 11, "Stage2b": 12, "LA": 0}
TREE_DEPTH = 5


def replace_na_with_parameter(data, label, param):
    m = param(data[label].dropna())
    return data[label].fillna(m)


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def find_num(inputString):
    s = ""
    first_i = len(inputString)
    has_dot = False
    for i, _ in enumerate(inputString):
        if _.isdigit() and first_i == len(inputString):
            first_i = i
        if _ == "." and has_dot is False:
            s += _
            has_dot = True
        if _.isdigit() or (_ == "-" and i < first_i):
            s += _
    return float(s)


def er_col(s):
    neg_list = ["neg", "begative", "-", "nge", "שלילי"]
    pos_list = ["yes", "+", "po", "high", "pr", "חיובי"]
    if isinstance(s, str):
        if has_numbers(s):
            return np.sign(find_num(s))
        for neg in neg_list:
            if neg in s:
                return -1
        for pos in pos_list:
            if pos.lower() in s.lower():
                return 1
        return 0
    return 0


def hist_diagnosis(s):
    neg_list = ["CARCINOMA", "MALIGNANT", "BENIGN", "ADENOCARCINOMA", "INTRADUCTAL PAPILLOMA", "PHYLLODES TUMOR NOS"]
    pos_list = ["MICROPAPILLARY VARIANT", "ADENO", "INTRADUCTAL", "PAGET`S DISEASE OF BREAST", "FIBROADENOMA"]
    if isinstance(s, float):
        return 0
    if isinstance(s, str):
        for neg in neg_list:
            if neg in s:
                return -1
        for pos in pos_list:
            if pos in s:
                return 1
        return 0


def her2(s):
    neg_list = ["neg", "Neg", "eg", "EG", "שלילי", "nec", "akhkh","akhah", "nef", "nfg", "ag", "nrg", "1"]
    pos_list = ["pos", "Pos", "POS", "3","חיובי", "2", "po"]
    if isinstance(s, float):
        return 0
    if isinstance(s, str):
        if "-" in s:
            return -1
        if "+" in s:
            return 1
        for neg in neg_list:
            if neg in s:
                return -1
        for pos in pos_list:
            if pos in s:
                return 1
        return 0


def ivi(s):
    neg_list = ["No", "no", "neg", "NO", "-", "not"]
    pos_list = ["pos", "yes", "+", "MICROPAPILLARY VARIANT"]
    if isinstance(s, float):
        return 0
    if isinstance(s, str):
        for neg in neg_list:
            if neg in s:
                return -1
        for pos in pos_list:
            if pos in s:
                return 1
        return 0


def ki67(s):
    if isinstance(s, str):
        s = s.replace("%","")
        if s.isdigit():
            i = int(s)
            if i > 100:
                return int(i/100)
            return int(i/10)


def Nnodes(s):
    if isinstance(s, float):
        return -1
    if isinstance(s, str):
        for i in range(5):
            if str(i) in s:
                return i
        return -1


def info(df, s):
    print(df[s].value_counts(), "\nnull values:", df[s].isna().sum(), "\n")


def flatten(ls):
    """
    flatten a nested list
    """
    flat_ls = list(itertools.chain.from_iterable(ls))
    return flat_ls


class Encode_Multi_Hot:
    """
    change the variable length format into a
    fixed size one hot vector per each label
    """

    def __init__(self):
        """
        init data structures
        """
        self.label_to_ind = {}
        self.ind_to_label = {}
        self.num_of_label = None

    def fit(self, raw_labels):
        """
        learn about possible labels
        """
        # get a list of unique values in df
        labs = list(set(flatten(raw_labels)))
        inds = list(range(len(labs)))
        self.label_to_ind = dict(zip(labs, inds))
        self.ind_to_label = dict(zip(inds, labs))
        self.num_of_label = len(labs)

    def enc(self, raw_label):
        """
        encode variable length category list into multiple hot
        """
        multi_hot = np.zeros(self.num_of_label)
        for lab in raw_label:
            cur_ind = self.label_to_ind[lab]
            multi_hot[cur_ind] = 1
        return multi_hot

    def decode(self, multi_hot):
        sorted_key = sorted(self.ind_to_label.keys())
        labels = []
        for k in sorted_key:
            labels.append(self.ind_to_label[k])
        output = []
        for vec in multi_hot:
            output.append(np.array(labels)[vec == 1])
        return output


def parse_df_labels(df):
    """
    Return a dictionary of response name and values from df
    """
    assert (len(df.columns) == 1)
    resp = df.columns[0]
    ls = [eval(val) for val in df[resp]]
    ret_dict = {"resp": resp, "vals": ls}
    return ret_dict


def preprocess_part1(feats_path, labels_path=None):
    # read csv
    df = pd.read_csv(feats_path, low_memory=False)
    if labels_path is not None:
        labels = pd.read_csv(labels_path)
        df = pd.concat([df, labels], axis=1)

    df['אבחנה-Basic stage'] = df['אבחנה-Basic stage'].map(
        {"Null": 0, "c - Clinical": 1, "p - Pathological": 2, "r - Reccurent": 3})
    # Her2
    df['אבחנה-Her2'] = df['אבחנה-Her2'].apply(her2)
    # margin type
    df['אבחנה-Margin Type'] = df['אבחנה-Margin Type'].map({"ללא": 0, 'נקיים': 1, 'נגועים': -1})
    # hist degree
    df['אבחנה-Histopatological degree'] = df['אבחנה-Histopatological degree'].map(
        {"Null": -1, "GX - Grade cannot be assessed": 0, "G1 - Well Differentiated": 1,
         "G2 - Modereately well differentiated": 2, "G3 - Poorly differentiated": 3,
         "G4 - Undifferentiated": 3})
    df['אבחנה-Ivi -Lymphovascular invasion'] = df['אבחנה-Ivi -Lymphovascular invasion'].apply(ivi)
    df['אבחנה-KI67 protein'] = df['אבחנה-KI67 protein'].apply(ki67)
    df['אבחנה-KI67 protein'] = df['אבחנה-KI67 protein'].fillna(-1)
    df['אבחנה-Lymphatic penetration'] = df['אבחנה-Lymphatic penetration'].map(
        {"Null": -1, "L0 - No Evidence of invasion": 0, "LI - Evidence of invasion": 1,
         "L1 - Evidence of invasion of superficial Lym.": 2, "L2 - Evidence of invasion of depp Lym.": 3})
    df['אבחנה-M -metastases mark (TNM)'] = df['אבחנה-M -metastases mark (TNM)'].map({"M0": -1, "M1": 1, "MX": 0, "Not yet Established": 0, "M1a": 1, "M1b": 1})
    df['אבחנה-M -metastases mark (TNM)'] = df['אבחנה-M -metastases mark (TNM)'].fillna(0)
    #TMN
    df['אבחנה-N -lymph nodes mark (TNM)'] = df['אבחנה-N -lymph nodes mark (TNM)'].apply(Nnodes)
    #Tumor mark
    df['אבחנה-T -Tumor mark (TNM)'] = df['אבחנה-T -Tumor mark (TNM)'].apply(Nnodes)

    # Noy and Shani
    df["אבחנה-Nodes exam"] = replace_na_with_parameter(df, "אבחנה-Nodes exam", np.median)
    df["אבחנה-Positive nodes"] = replace_na_with_parameter(df, "אבחנה-Positive nodes",np.median)
    df["left"] = (df["אבחנה-Side"] == "שמאל") | (df["אבחנה-Side"] == "דו צדדי")
    df["right"] = (df["אבחנה-Side"] == "ימין") | (df["אבחנה-Side"] == "דו צדדי")
    df["אבחנה-Stage"] = pd.DataFrame(list(map(lambda x: sorted_dict[x] if x in sorted_dict else -1, df["אבחנה-Stage"])))
    df["אבחנה-Surgery sum"] = replace_na_with_parameter(df, "אבחנה-Surgery sum",lambda x: 0)
    df["אבחנה-Tumor width"] = replace_na_with_parameter(df, "אבחנה-Tumor width", lambda x: x if type(x) in {int,float} else -1)
    df["אבחנה-er"] = pd.DataFrame(list(map(er_col, df["אבחנה-er"])))
    df["אבחנה-pr"] = pd.DataFrame(list(map(er_col, df["אבחנה-pr"])))
    df["אבחנה-Histological diagnosis"] = pd.DataFrame(list(map(er_col, df["אבחנה-Histological diagnosis"])))
    # create dummys for tumor width
    for i in range(9):
        df[f"Tumor width in {i}-{i + 1}"] = (i <= df["אבחנה-Tumor width"]) & (df["אבחנה-Tumor width"] < i + 1)
    # check positive values
    for feat in CHECK_POS:
        df = df[df[feat] >= 0]
    # drop features
    df = df.drop(labels=[" Form Name"], axis=1)
    if labels_path is not None:
        df = df.drop_duplicates(["id-hushed_internalpatientid"])
    df = df.drop(TO_REMOVE, axis=1)
    return df


def main(argv):
    np.random.seed(0)
    # part 1
    # 1.Load dataset from the source.
    features_path, labels1_path, labels2_path, test_features_path = argv
    response_txt1, response_txt2 = "אבחנה-Location of distal metastases", "אבחנה-Tumor size"
    df = preprocess_part1(features_path, labels1_path)
    y = df.pop(response_txt1)
    y = pd.DataFrame(y)
    X_train = df
    X_test = preprocess_part1(test_features_path)

    # 2.Train Decision tree, SVM, and KNN classifiers on the training data.

    encode = Encode_Multi_Hot()
    classes = parse_df_labels(y)["vals"]
    encode.fit(classes)
    labels_mult_vec = [encode.enc(val) for val in classes]
    forest = RandomForestClassifier(random_state=1)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=2)
    multi_target_forest.fit(X_train, labels_mult_vec)
    pred = multi_target_forest.predict(X_test)
    pred = encode.decode(pred)
    # export to csv
    with open("part1/predictions.csv", 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([np.array(response_txt1)])
        for val in pred:
            writer.writerow([list(val)])

    # part 2
    df = preprocess_part1(argv[0], argv[2])
    y = df.pop(response_txt2)
    X_train = df

    # Ridge with cross validation
    rrcv = RidgeCV(store_cv_values=True, alphas=np.linspace(0.1, 7, 100))
    rrcv.fit(X_train, y)
    ridge_pred = rrcv.predict(X_test)
    # Graph
    # px.line(x=list(np.linspace(0.1, 7, 100)), y=np.mean(rrcv.cv_values_, axis=0)).show()
    # export to csv
    with open("part2/predictions.csv", 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([np.array(response_txt2)])
        for val in ridge_pred:
            writer.writerow([val])


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Wrong Usage please use like this <train.features.csv path> <labels.0.csv path> <labels.1.csv path> "
              "<test.features.csv path>")
    main(sys.argv[1:])
