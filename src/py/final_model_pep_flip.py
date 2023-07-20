import os
import numpy as np
import pandas as pd
from scipy import stats

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import yellowbrick as yb
from matplotlib.colors import ListedColormap
from yellowbrick.classifier import ROCAUC
from matplotlib_venn import venn3
import matplotlib.patches as mpatches
# Statistics, EDA, metrics libraries
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.utils import class_weight
from sklearn.compose import ColumnTransformer

# Modeling libraries
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict,KFold, cross_validate
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import roc_curve, auc
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, EditedNearestNeighbours
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.metrics import SCORERS
from xgboost import XGBClassifier
import pickle


RANDOM_STATE = 42


def split_data(data):

    classes = ['pep_flip', 'random']
    data = data[data.conformation_type.isin(classes)]


    data.reset_index(drop=False, inplace=True)

    residue_info = data[['asymID','compID', 'insCode', 'seqNum', 'seqID_besttls','index']]
    conformation_type = data["conformation_type"]
    #df = data.drop(["conformation_type",'asymID','compID', 'insCode', 'seqNum', 'seqID_besttls','index'], axis=1)
    #df = data.loc[:, 'cn_m1':'dcaca']
    df = data.loc[:, 'cn_m1':'dssp_p2T']
    labelencoder = LabelEncoder()

    conformation_type_encoded = pd.Series(labelencoder.fit_transform(conformation_type), name = "encoded_label")
    
    label = pd.concat([conformation_type, conformation_type_encoded], axis=1)

    X = df
    y = label["encoded_label"].array

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = RANDOM_STATE  , stratify=y)


    return X_train, X_test, y_train, y_test


def get_final_model(X_train,y_train):

    to_scale = np.arange(0,80).tolist()
    
    data_transform = ColumnTransformer(
        transformers = 
            [ ("numerical", StandardScaler(), to_scale)] ,
        remainder = 'passthrough'
    )

    kbest = SelectKBest(f_classif)
    
    clf1 = KNeighborsClassifier()
    clf2 = RandomForestClassifier(class_weight = "balanced")
    clf3 = XGBClassifier()
    clf4 = SVC(probability=True, class_weight= "balanced")
    
    pipe = Pipeline(steps= [('kbest', kbest) ,('classifier', clf1)] )
    
    n_neighbors_enn = np.arange(1,11).tolist()

    params1 = {
                'kbest__k': [20,40,60,80,100],
                'classifier__n_neighbors': (1, 10, 1),
                'classifier__leaf_size': (20, 40, 1),
                'classifier__p': (1, 2),
                'classifier__weights': ('uniform', 'distance'),
                'classifier__metric': ('minkowski', 'chebyshev'),
                'classifier' :[clf1]
    }
    

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 20, stop = 600, num = 20)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    #sampling strategy for smote
    

    # Create the random grid
    params2 = {
                'kbest__k': [20,40,60,80,100],
                'classifier__n_estimators': n_estimators,
                'classifier__max_features': max_features,
                'classifier__max_depth': max_depth,
                'classifier__min_samples_split': min_samples_split,
                'classifier__min_samples_leaf': min_samples_leaf,
                'classifier__bootstrap': bootstrap,
                'classifier' :[clf2]
                    }
    
    

    params3 = { 
                'kbest__k': [20,40,60,80,100],
                'classifier__min_child_weight': [1, 5, 10, 100],
                'classifier__gamma': [0.5, 1, 1.5, 2, 5],
                'classifier__subsample': [0.6, 0.8, 1.0],
                'classifier__colsample_bytree': [0.6, 0.8, 1.0],
                'classifier__max_depth': [3, 6, 9],
                'classifier__learning_rate':[0.05, 0.1, 0.20],
                'classifier__n_estimators': [100, 400, 800],
                'classifier__eval_metric': ['merror', 'mlogloss'],
                'classifier__objective': ['binary:logistic', 'binary:hinge'],
                'classifier' :[clf3]

        }
    
    params4 = {
                'kbest__k': [20,40,60,80,100],
                'classifier__C': [0.1,1,100,1000],
                'classifier__kernel': ['linear'],
                'classifier__gamma': [0.001, 0.01, 0.1, 1, 'auto', 'scale'],
                'classifier__degree':[1,2,3,4,5,6],
                'classifier' :[clf4]
    }
    
    params = [params1, params2, params3]
    
    scoring = {"AUC" : "roc_auc", "Accuracy": make_scorer(accuracy_score), "F1_score": 'f1', 'recall': 'recall', 'balanced_accuracy':'balanced_accuracy'}
    
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    
    clf = RandomizedSearchCV(pipe, param_distributions=params, n_jobs = -10 ,verbose= 2, cv=inner_cv, scoring = scoring, return_train_score= True, refit="balanced_accuracy", error_score="raise")

    clf.fit(X_train,y_train)
   
    return clf

def run_script():

    data = pd.read_csv("data/training_data/final_training_set.csv")
    
    X_train, X_test, y_train, y_test = split_data(data)

    """class_weights = class_weight.compute_class_weight(class_weight = "balanced",
                                                  classes= np.unique(y_train), 
                                                  y= y_train )


    dict_weights = dict(zip(np.unique(y_train), class_weights))"""
    
    clf = get_final_model(X_train,y_train) 
    
    clf.fit(X_train,y_train)

    with open('provisional_pep_flip.pkl', 'wb') as files:
        pickle.dump(clf, files)

    y_pred = clf.predict(X_test)
    
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    print(f"balanced_accuracy final_model: {balanced_accuracy}")

    #get best selected features
    fs = clf.best_estimator_.named_steps['kbest']
    features = np.array(X_train.columns)
    selected_features = features[fs.get_support()]
    print("Features selected: ", selected_features)
    print("\n The best parameters across ALL searched params:\n", clf.best_params_)

    print(classification_report(y_test, y_pred))
    scores = fs.scores_

    """#bar plot of all selected feature with their correspondant F-score
    features = list(map(str, selected_features))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(features, scores)
    plt.show()"""


    

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    # Create the matrix
    cm = confusion_matrix(y_test, y_pred)
    cmp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
    cmp.plot(ax=ax)

    plt.show();
    
    
    
def main():

    run_script()

if __name__ == '__main__':
    main()