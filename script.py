
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import sklearn
import joblib
import boto3
import pathlib
from io import StringIO
import argparse
import numpy as np
import pandas as pd
import os

def model_fn(model_dir):
    clf=joblib.load(os.path.join(model_dir,"model.joblib"))
    return clf
if __name__=="__main__":
    print('extracting arguments')
    parser=argparse.ArgumentParser()



    parser.add_argument("--model-dir",type=str,default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train',type=str,default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test',type=str,default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file',type=str,default='trainX-v1.csv')
    parser.add_argument('--test-file',type=str,default='testX-v1.csv')
    parser.add_argument("--n_estimators", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=0)


    args, _=parser.parse_known_args()

    print('Sk learn version: ',sklearn.__version__)
    print('Sk learn version: ',joblib.__version__)

    print('[INFO] reading data')

    print()

    train_df=pd.read_csv(os.path.join(args.train,args.train_file))
    test_df=pd.read_csv(os.path.join(args.test,args.test_file))

    features=list(train_df.columns)
    label=features.pop(-1)

    print('Building training and testing datasets')

    print()

    X_train=train_df[features]
    X_test=test_df[features]
    y_train=train_df[label]
    y_test=test_df[label]

    print('Column order: ')
    print(features)
    print()

    print('Label column is: ',  label)
    print()

    print('Data shape: ')
    print()

    print('--- SHAPE of training data 85%---')
    print(X_train.shape)
    print(y_train.shape)
    print()

    print('SHApe of testing data---')


    print(X_test.shape)
    print(y_test.shape)
    print()


    print('TRaining random model')

    print()

    model=RandomForestClassifier(n_estimators=args.n_estimators,random_state=args.random_state)
    model.fit(X_train,y_train)
    print()

    model_path=os.path.join(args.model_dir,"model.joblib")
    joblib.dump(model,model_path)
    print("Model persisted at",model_path)
    print()

    y_pred_test=model.predict(X_test)
    test_acc=accuracy_score(y_test,y_pred_test)
    test_rep=classification_report(y_test,y_pred_test)

    print()

    print('--testing results for testing data-----')

    print()
    print('total rows are: ',X_test.shape[0])
    print('[Testing] model accuracy is: ',test_acc)
    print('[Testing] testing report: ')
    print(test_rep)
