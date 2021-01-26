# ---------- Decision Trees ------------ #

import pandas as pd
from sklearn.tree import DecisionTreeClassifier #
from sklearn.model_selection import train_test_split #helps with splitting data
from sklearn import metrics #will help with calculating accuracy

def decision_tree_sklearn(df,
                          target_column,
                         feature_columns=None,
                          perform_testing=False,
                          test_size = 0,
                          
                          # parameters for the decision tree
                            criterion = "entropy", # entropy for infromation gain, gini fro gini index
                            splitter = "best", #For the splitt strategy,also can be "random"
                            max_depth = None,
                            max_features = None,
                            min_samples_split= 0.1,
                            min_samples_leaf = 0.02,
                          
                         ):
    
    """
    Purpose: To train a decision tree
    based on a dataframe with the features and the classifications
    
    Parameters:
    max_depth = If None then the depth is chosen so all leaves contin less than min_samples_split
                            The higher the depth th emore overfitting


    
    """
    
    if feature_columns is None:
        feature_columns = list(df.columns)
        feature_columns.remove(target_column)
        
    
    col_names = feature_columns
    
    #1) dividing out the data into features and classifications
    X = df[col_names]
    y = df[[target_column]]
    
    
    #2) Dividing the data into test and training set
    test_size = 0
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=1)
    else:
        X_train = X_test = X
        y_train = y_test = y

        
        
    
    #3) Train the classifier
    clf = DecisionTreeClassifier(criterion=criterion,
                                splitter=splitter,
                                max_depth = max_depth,
                                max_features=max_features,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf
                                )

    # training the model
    clf = clf.fit(X_train,y_train)

    
    if perform_testing and test_size > 0:
        #testing the trained model
        y_pred = clf.predict(X_test)

        #measure the accuracy
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        
    return clf

# How to visualize tree 
from IPython.display import SVG
from graphviz import Source
from IPython.display import display   

from sklearn.tree import export_graphviz

def plot_decision_tree(clf,
                      feature_names,
                      class_names=None):
    """
    Purpose: Will show the 
    
    
    """
    if class_names is None:
        class_names = list(clf.classes_)
    
    graph = Source(export_graphviz(clf
      , out_file=None
      , feature_names=feature_names
      , class_names=class_names
      , filled = True
      , precision=6))

    print(f"clf.classes_ = {clf.classes_}")
    display(SVG(graph.pipe(format='svg')))
    
    
import numpy as np
def print_tree_structure_description(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    
    
#     if feature_names is None:
#         feature = clf.tree_.feature
#     else:
#         feature = feature_names
        
        
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has {n} nodes and has "
          "the following tree structure:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}node={node} is a leaf node.".format(
                space=node_depth[i] * "\t", node=i))
        else:
            print("{space}node={node} is a split node: "
                  "go to node {left} if X[:, {feature}] <= {threshold} "
                  "else to node {right}.".format(
                      space=node_depth[i] * "\t",
                      node=i,
                      left=children_left[i],
                      feature=feature[i],
                      threshold=threshold[i],
                      right=children_right[i]))
