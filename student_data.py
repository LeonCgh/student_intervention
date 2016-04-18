import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn import grid_search
import matplotlib.pyplot as pl


student_data = pd.read_csv("C:\Users\Chen\Downloads\student_intervention (3)\student_intervention\student-data.csv")
print "Student data read successfully!"


     


n_students = student_data.shape[0]
n_features = student_data.shape[1]
n_passed = student_data.as_matrix(columns=[student_data.columns[-1]]).tolist().count(['yes'])
n_failed = student_data.as_matrix(columns=[student_data.columns[-1]]).tolist().count(['no'])
grad_rate = n_passed/n_passed+n_failed
print "Total number of students: {}".format(n_students)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Number of features: {}".format(n_features)
#print "Graduation rate of the class: {:.2f}%".format(grad_rate)


# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows


# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(X_all)
print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))


# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, test_size=0.3, train_size=0.7,random_state=0)
print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])
# Note: If you need a validation set, extract it from within training data

import time

def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

clf_S = svm.SVC()
clf_T= RandomForestClassifier(n_estimators=10)
clf_G=GaussianNB()

# Fit model to training data
model1=train_classifier(clf_S, X_train, y_train)  # note: using entire training set here
#print clf  # you can inspect the learned model by printing it
model2=train_classifier(clf_T, X_train, y_train)

model3=train_classifier(clf_G, X_train, y_train)


def predict_labels(clf, features, target):
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')

train_f1_score_svc = predict_labels(clf_S, X_train, y_train)
train_f1_score_Tree = predict_labels(clf_T, X_train, y_train)
train_f1_score_Bayes = predict_labels(clf_G, X_train, y_train)


print "F1 score for training set: {}".format(train_f1_score_svc)
print "F1 score for training set: {}".format(train_f1_score_Tree)
print "F1 score for training set: {}".format(train_f1_score_Bayes)

print "F1 score for test set: {}".format(predict_labels(clf_S, X_test, y_test))
print "F1 score for test set: {}".format(predict_labels(clf_T, X_test, y_test))
print "F1 score for test set: {}".format(predict_labels(clf_G, X_test, y_test))

def train_predict(clf, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    print "Training set size: {}".format(len(X_train))
    train_classifier(clf, X_train, y_train)
    print "F1 score for training set: {}".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))

train_predict(clf_S, X_train[:100], y_train[:100], X_test, y_test)
train_predict(clf_S, X_train[:200], y_train[:200], X_test, y_test)
train_predict(clf_S, X_train, y_train, X_test, y_test)

train_predict(clf_T, X_train[:100], y_train[:100], X_test, y_test)
train_predict(clf_T, X_train[:200], y_train[:200], X_test, y_test)
train_predict(clf_T, X_train, y_train, X_test, y_test)

train_predict(clf_G, X_train[:100], y_train[:100], X_test, y_test)
train_predict(clf_G, X_train[:200], y_train[:200], X_test, y_test)
train_predict(clf_G, X_train, y_train, X_test, y_test)



def performance_metric(y_test, y_pred):
    

    return f1_score(y_test, y_pred, pos_label='yes')
    
"""
def tree_learning_curve(depth, X_train, y_train, X_test, y_test):
   # Calculate the performance of the model after a set of training data.

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.round(np.linspace(1, len(X_train), 50))
    sizes=sizes.astype(int)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth: "
    print depth

    for i, s in enumerate(sizes):

        # Create and fit the decision tree regressor model
        regressor = RandomForestClassifier(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))


    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err)


def learning_curve_graph(sizes, train_err, test_err):
    #Plot training and test error as a function of the training size.

    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    pl.plot(sizes, test_err, lw=2, label = 'test error')
    pl.plot(sizes, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.show()

def fit_model(X, y):
    # Tunes a decision tree regressor model using GridSearchCV on the input data X 
        and target labels y and returns this optimal model.
        
    start = time.time()

    # Create a decision tree regressor object
    regressor = RandomForestClassifier()

    # Set up the parameters we wish to tune
    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    # Make an appropriate scoring function
    scoring_function = make_scorer(performance_metric)

    # Make the GridSearchCV object
    reg = RandomizedSearchCV(regressor, param_grid=parameters,scoring=scoring_function)

    # Fit the learner to the data to obtain the optimal model with tuned parameters
    reg.fit(X, y)
    
    end=start = time.time()
    
    print "Done!\nfit time (secs): {:.3f}".format(end - start)

    # Return the optimal model
    print("The best parameters are %s with a score of %0.2f"
      % (reg.best_params_, reg.best_score_))
    return reg.best_params_
    
max_depths = [1,2,3,4,5,6,7,8,9,10]
for max_depth in max_depths:
    tree_learning_curve(max_depth, X_train, y_train, X_test, y_test)
    
    
def model_complexity(X_train, y_train, X_test, y_test):
   #Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. 
    
    print "Creating a model complexity graph. . . "

    # We will vary the max_depth of a decision tree model from 1 to 14
    max_depth = np.arange(1, 14)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = RandomForestClassifier(max_depth = d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    pl.figure(figsize=(7, 5))
    pl.title('Decision Tree Classifier Complexity Performance')
    pl.plot(max_depth, test_err, lw=2, label = 'Testing Error')
    pl.plot(max_depth, train_err, lw=2, label = 'Training Error')
    pl.legend()
    pl.xlabel('Maximum Depth')
    pl.ylabel('Total Error')
    pl.show()


model_complexity(X_train, y_train, X_test, y_test)


def learning_curve(regressor,X_train, y_train, X_test, y_test):
    #Calculate the performance of the model after a set of training data.

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.round(np.linspace(10, len(X_train), 50))
    sizes=sizes.astype(int)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    

    for i, s in enumerate(sizes):

        # Create and fit the decision tree regressor model
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))


    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err)



learning_curve(clf_S,X_train, y_train, X_test, y_test)

learning_curve(clf_G,X_train, y_train, X_test, y_test)
"""

    
    

    



def fit_model_svm(X, y):
    """ Tunes a decision tree regressor model using GridSearchCV on the input data X 
        and target labels y and returns this optimal model. """
        
    start = time.time()
    
    C_range = np.logspace(-10, 10, 10)
    gamma_range = np.logspace(-10, 10, 10)
    kernel_range= np.logspace(-10, 10, 10)
    scoring_function = make_scorer(performance_metric)
    #param_grid = dict(gamma=gamma_range, C=C_range,Kernel=kernel_range)
    param_grid = [{'gamma':gamma_range, 'C':C_range,'kernel': ['rbf'], 'tol':kernel_range}]
    reg = grid_search.GridSearchCV(svm.SVC(), param_grid=param_grid,scoring=scoring_function)
    reg.fit(X, y)
    
    end = time.time()
    
    print "Done!\nfit time (secs): {:.3f}".format(end - start)
    
    print("The best parameters are %s with a score of %0.2f"
      % (reg.best_params_, reg.best_score_))
    return reg

"""def fit_model_NB(X, y):
     Tunes a decision tree regressor model using GridSearchCV on the input data X 
        and target labels y and returns this optimal model.
    
  start = time.time()
    
    #y= np.linspace(0, 1, 20)
    #u = np.linspace(0, 1, 20)
  scoring_function = make_scorer(performance_metric)
    #param_grid = dict(y=y, u=u)
   reg = GridSearchCV(GaussianNB(), param_grid=None,scoring=scoring_function)
   reg.fit(X, y)
    
  end = time.time()
    
    print "Done!\nfit time (secs): {:.3f}".format(end - start)
    
    print("The best parameters are %s with a score of %0.2f"
      % (reg.best_params_, reg.best_score_))
    return reg
"""

"""def Navie_fit_model(X_train,y_train,X_all,y_all):
    regressor=GaussianNB()
    regressor.fit(X_train,y_train)
    score=f1_score(y_all.values, regressor.predict(X_all), pos_label='yes')
    
    return score
    
    
    
    
fit_model(X_all,y_all)    
"""

fit_model_svm(X_all,y_all)





    
    
    
    
    