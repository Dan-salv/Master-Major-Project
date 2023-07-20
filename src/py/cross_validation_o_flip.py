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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.utils import class_weight
from sklearn.compose import ColumnTransformer

# Modeling libraries
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict,KFold, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import roc_curve, auc
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, EditedNearestNeighbours, RandomUnderSampler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.metrics import SCORERS
from xgboost import XGBClassifier
import json

RANDOM_STATE = 42


def split_data(data):

    data = pd.read_csv("data/training_data/final_training_set.csv")

    classes = ['o_flip', 'random']
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = RANDOM_STATE  , stratify=y)

    return X_train, X_test, y_train, y_test

def get_params():

    os_strategy = [
    0.015, 0.02
    ]
    us_strategy = [0.4 ,0.5, 0.6]
    n_neighbors_enn = np.arange(1,11).tolist()

    params1 = {
                'kbest__k': [20,40,60,80,100],
                'knn__n_neighbors': (1, 10, 1),
                'knn__leaf_size': (20, 40, 1),
                'knn__p': (1, 2),
                'knn__weights': ('uniform', 'distance'),
                'knn__metric': ('minkowski', 'chebyshev'),
                "om__sampling_strategy": os_strategy,
                "um__sampling_strategy": us_strategy
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
                'rf__n_estimators': n_estimators,
                'rf__max_features': max_features,
                'rf__max_depth': max_depth,
                'rf__min_samples_split': min_samples_split,
                'rf__min_samples_leaf': min_samples_leaf,
                'rf__bootstrap': bootstrap,
                "om__sampling_strategy": os_strategy,
                "um__sampling_strategy": us_strategy,
                #'enn__n_neighbors': n_neighbors_enn
                    }
    
    params3 = {
                'kbest__k': [20,40,60,80,100],
                'svm__C': [0.1,1,100,1000],
                'svm__kernel': ['linear'],
                'svm__gamma': [0.001, 0.01, 0.1, 1, 'auto', 'scale'],
                'svm__degree':[1,2,3,4,5,6],
                "om__sampling_strategy": os_strategy,
                "um__sampling_strategy": us_strategy,
    }

    params4 = { 
                'kbest__k': [20,40,60,80,100],
                'xgb__min_child_weight': [1, 5, 10, 100],
                'xgb__gamma': [0.5, 1, 1.5, 2, 5],
                'xgb__subsample': [0.6, 0.8, 1.0],
                'xgb__colsample_bytree': [0.6, 0.8, 1.0],
                'xgb__max_depth': [3, 6, 9],
                'xgb__learning_rate':[0.05, 0.1, 0.20],
                'xgb__n_estimators': [100, 400, 800],
                'xgb__eval_metric': ['merror', 'mlogloss'],
                'xgb__objective': ['binary:logistic', 'binary:hinge'],
                'xgb__scale_pos_weight': [1, 10, 25, 50],
                "om__sampling_strategy": os_strategy,
                "um__sampling_strategy": us_strategy

        }
   
    return params1, params2, params3, params4

def nested_crossvalidation(model,param,X , Y):

    scoring = {"AUC" : "roc_auc", "Accuracy": make_scorer(accuracy_score), "F1_score": 'f1', 'recall': 'recall', 'balanced_accuracy':'balanced_accuracy'}
    
    estimators = {}
    num_trial = 5
    nested_score_acc = np.zeros(num_trial)
    nested_score_AUC = np.zeros(num_trial)
    nested_score_F1 =  np.zeros(num_trial)
    nested_score_recall= np.zeros(num_trial)
    nested_score_balanced_accuracy =  np.zeros(num_trial)

    nested_train_acc = np.zeros(num_trial)
    nested_train_AUC = np.zeros(num_trial)
    nested_train_F1 =  np.zeros(num_trial)
    
    #Run each trial 10 times
    for i in range(num_trial):
        #inner loop of 10 folds for hyperparameter tuning and feature selection
        inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
        #outer loop of 3 folds for cross validation
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        #randomize search for all parameters and number of optimal features
        clf = RandomizedSearchCV(model, param_distributions=param, n_jobs = -10 , cv=inner_cv, scoring = scoring, verbose=2, return_train_score= True, refit="balanced_accuracy", error_score="raise")
        #get cross-validation scores
        nested_score = cross_validate(clf, X=X, y=Y, scoring=scoring, cv=outer_cv, return_train_score=True, return_estimator=True, error_score="raise")

        #get best parameter for all 10 estimators
        estimators_i = []

        print(nested_score)
        print(nested_score["estimator"])
        
        print("CV accuracy scores: ", nested_score["test_Accuracy"])
        print("CV AUC scores: ",nested_score["test_AUC"])
        print("CV F1 scores: ",nested_score["test_F1_score"])
        print("CV recall scores: ",nested_score["test_recall"])
        print("CV balanced_accuracy scores: ",nested_score["test_balanced_accuracy"])
        
        print(f'Trial number {i} finished')
        """
        for estimator in nested_score["estimator"]:
            estimators_i.append(estimator.best_params_)
        estimators[i] = estimators_i"""

        #get metrics of accuracy scores and AUC for each classifier
        nested_score_acc[i] = nested_score["test_Accuracy"].mean()
        nested_score_AUC[i] = nested_score["test_AUC"].mean()
        nested_score_F1[i] = nested_score["test_F1_score"].mean()
        nested_score_recall[i] = nested_score["test_recall"].mean()
        nested_score_balanced_accuracy[i] = nested_score["test_balanced_accuracy"].mean()

    print("CV all accuracy scores: ", nested_score_acc);
    print("CV all AUC scores: ",nested_score_AUC)
    print("CV all F1 scores: ",nested_score_F1)
    print("CV all recall scores: ",nested_score_recall)
    print("CV all balanced_accuracy scores: ",nested_score_balanced_accuracy)

    scores = {"acc":nested_score_acc, "AUC":nested_score_AUC, "F1":nested_score_F1, "recall":nested_score_recall,"balanced_accuracy":nested_score_balanced_accuracy}

    return scores

def print_scores(scores):

    #models = ["KNN", "RFC", "SVM", "XGB"]
    models = ["KNN", "RFC","XGB"]
    model_dict={}

    for score, model in zip(scores, models):
        if not model in model_dict:
            model_dict[model] = {"mean":{},"std":{}}
        model_dict[model]["mean"] = {k:np.mean(v) for k, v in score.items()}
        model_dict[model]["std"] = {k:np.std(v) for k, v in score.items()}
        model_dict[model]["scores"] = {k:list(v) for k, v in score.items()}
    with open("data/training_data/scores/training_scores_o_flip.json", "w") as outfile:
        json.dump(model_dict, outfile)




def run_script():

    data = pd.read_csv("data/training_data/final_training_set.csv")
    
    X_train, X_test, y_train, y_test = split_data(data)

    """class_weights = class_weight.compute_class_weight(class_weight = "balanced",
                                                  classes= np.unique(y_train), 
                                                  y= y_train )


    dict_weights = dict(zip(np.unique(y_train), class_weights))"""
    
    to_scale = np.arange(0,80).tolist()
    
    data_transform = ColumnTransformer(
        transformers = 
            [ ("numerical", StandardScaler(), to_scale)] ,
        remainder = 'passthrough'
    )

    kbest = SelectKBest(f_classif)
    #under = EditedNearestNeighbours()
    under = RandomUnderSampler(random_state = RANDOM_STATE)
    over = RandomOverSampler(random_state = RANDOM_STATE)

    #Machine learning models
    KNN = KNeighborsClassifier()
    RFC = RandomForestClassifier(class_weight = "balanced")
    SVM = SVC(probability=True, class_weight= "balanced")
    XGB = XGBClassifier()

    #sme = SMOTEENN(random_state = RANDOM_STATE,smote= smote_sampler, enn= enn_sampler)
    model1 = Pipeline(steps= [("om", over) , ("um", under ), ('kbest', kbest) ,('knn', KNN)] )
    model2 = Pipeline(steps= [("om", over) , ("um", under ), ('kbest', kbest) ,('rf', RFC)] )
    model3 = Pipeline(steps= [("om", over) , ("um", under ),("preprocess", data_transform),('kbest', kbest) ,('svm', SVM)] )
    model4 = Pipeline(steps= [("om", over) , ("um", under) , ('kbest', kbest) ,('xgb', XGB)] )


    params1, params2, params3, params4 = get_params()
    
    #svm_scores = nested_crossvalidation(model3,params3, X_train, y_train)
    xgb_scores = nested_crossvalidation(model4,params4, X_train, y_train)
    knn_scores = nested_crossvalidation(model1,params1, X_train, y_train)
    rfc_scores = nested_crossvalidation(model2,params2, X_train, y_train)
    
    scores = [knn_scores,rfc_scores,xgb_scores]
    #scores = [knn_scores,rfc_scores,svm_scores,xgb_scores]
    print_scores(scores)
    
    
    
def main():

    run_script()

if __name__ == '__main__':
    main()