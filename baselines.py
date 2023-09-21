import pandas as pd
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

from tqdm import tqdm
from statistics import mean, stdev
dataset = pd.read_excel('~/downloads/GoalData_Scoring[1](1).xlsx')
dataset = dataset.dropna()

for column in dataset.columns[1:]:
    print(column)
    precs = []
    recs = []
    micro_f1s = []
    macro_f1s = []
    qwks = []
    for n in tqdm(range(5)):
        labels = dataset[column].values
        values = dataset[dataset.columns[0]].values
        kfold = StratifiedKFold(n_splits=5, shuffle=True)

        for train_index, val_index in kfold.split(X=values, y=labels):
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,5), max_df=0.95, min_df=0.01, max_features=5000)

            train_in = values[train_index]
            train_la = labels[train_index]
            test_in = values[val_index]
            test_la = labels[val_index]

            X_train = vectorizer.fit_transform(train_in)
            X_test = vectorizer.transform(test_in)

            classifier = RandomForestClassifier(n_estimators=300)
            classifier.fit(X_train, train_la)

            pred = classifier.predict(X_test)

            precs.append(precision_score(test_la, pred, average='micro'))
            recs.append(recall_score(test_la, pred, average='micro'))
            micro_f1s.append(f1_score(test_la, pred, average='micro'))
            macro_f1s.append(f1_score(test_la, pred, average='macro'))
            qwks.append(cohen_kappa_score(test_la, pred, weights='quadratic'))
    print(column)
    print(f'Precision - Mean: {mean(precs)} SD: {stdev(precs)} Min: {min(precs)} Max: {max(precs)}')
    print(f'Recall - Mean: {mean(recs)} SD: {stdev(recs)} Min: {min(recs)} Max: {max(recs)}')
    print(f'F1 Micro - Mean: {mean(micro_f1s)} SD: {stdev(micro_f1s)} Min: {min(micro_f1s)} Max: {max(micro_f1s)}')
    print(f'F1 Macro - Mean: {mean(macro_f1s)} SD: {stdev(macro_f1s)} Min: {min(macro_f1s)} Max: {max(macro_f1s)}')
    print(f'QWK - Mean: {mean(qwks)} SD: {stdev(qwks)} Min: {min(qwks)} Max: {max(qwks)}')


            

    



