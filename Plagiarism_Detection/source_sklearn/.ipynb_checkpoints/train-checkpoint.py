from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.externals import joblib

## TODO: Import any additional libraries you need to define a model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    GridSearchCV
)

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    

    ## TODO: Define a model 
    model = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=n_jobs,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None
    )
    
    params = dict(
        n_estimators=list(range(20, 220, 20)),
        max_depth=list(range(4, 44, 4))
    )
    
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

    grid_rf = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring={
            'accuracy': make_scorer(accuracy_score),
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'neg_log_loss': make_scorer(
                log_loss, needs_proba=True, labels=np.unique(y[:, i])
                )
        },
        refit='balanced_accuracy',
        cv=rskf,
        return_train_score=True,
        n_jobs=4,
        verbose=0)
    
    ## TODO: Train the model
    grid_rf.fit(train_x, train_y)
    model = grid_rf.best_estimator_

    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))