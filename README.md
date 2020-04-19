# ID3_decisiontree

In this repository an ID3 decision tree classifier is designed to classify the categorical dataset “Tic-Tac-Toe Endgame” and numerical dataset “Wine” from the UCI Machine Learning Repository. The project is discretized into the following sections:
1. Design of an ID3 decision tree classifier reporting accuracy based on 10-times-10-fold cross validation.
    1. Design an ID3 decision tree classifier using Information Gain for the constrution of the tree for both datasets.
    1. Repeat Part 1 using Gain-Ratio instead of Information Gain. 

1. Analysis of attribute noise and class noise on ID3 decision tree classifier.

## Part 1
An ID3 decision tree classifier based on information gain was developed in Python to classify the
datasets “Tic-Tac-Toe Endgame” and “Wine”. The mean and variance of the accuracy based on 10-
times-10-fold cross-validation are reported in Table I. 
TABLE I: ACCURACY RESULTS ID3 INFORMATION GAIN CLASSIFIER TREE 
|   Dataset   |     Criteria     | Accuracy - Mean | Accuracy - STD |
|:-----------:|:----------------:|:---------------:|:--------------:|
| Tic_Tac_Toe | Information Gain |      84.88%     |      3.49      |
|     Wine    | Information Gain |      91.15%     |      7.04      |

The confusion matrix for each dataset is shown in Figure 1 and Figure 2 for the Tic-Tac-Toe and
Wine dataset, respectively. 

![](FIGURES/TTT_IG_conf_mat2.png)  *Figure 1. Confusion matrix generated from Information Gain based ID3 tree for Tic Tac Toe dataset using 10 times 10-fold cross-validation. *  
![](FIGURES/IG_wine_conf.png)  *Figure 2. Confusion matrix generated from Information Gain based ID3 tree for Wine dataset using 10 times 10-fold cross-validation. *  
