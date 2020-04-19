#SYDE 675
#Matthew Cann
#20863891
#Assignment 2 Question 2 - Discrete


#IMPORTS.......................................................................
import pandas as pd
from numpy import log2 as log
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os # accessing directory structure
import numpy as np
import matplotlib.pyplot as plt


eps = np.finfo(float).eps #Small value such that log won't fail

Directory = "E:\Documents\Waterloo-Masters\SYDE 675\Assignment2"

os.chdir(Directory) 


#%% DATASETS...................................................................

def get_tic_tac_toe_dataset():
    df = pd.read_csv('tic-tac-toe.data', delimiter=',')

    df.columns=['top-left-square','top-middle-square','top-right-square', 'middle-left-square', 
              'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'Class']
    #df.to_excel("outputttt.xlsx")

    return df

#%% LOAD DATASET...............................................................
df = get_tic_tac_toe_dataset()
class_attribute  = 'Class'

print(df)
    
#%%FUNCTIONS...................................................................

def accuracy(y_true, y_predicted):
    '''Reports the accuracy of two lists of true and predicted values'''
    correct = 0
    for true, pred in zip(y_true, y_predicted):        
        if str(true) == str(pred):
            correct += 1
    accuracy = correct/len(y_predicted)*100
    print('Accuracy of classifer {:0.2f} %' .format(accuracy))
    return accuracy

def entropy(dataframe, target_attribute):
    '''Calculates the Entropy of a dataset for the target attribute'''
    entropy = 0  #Initialize Entropy
    values = dataframe[target_attribute].unique() #Play has two options 'Yes', 'No'
    
    play_data = list(dataframe[target_attribute].values)

    for value in values:
        proportion = play_data.count(value)/len(play_data) #Proportion of given value in the dataset
        entropy += -proportion*log(proportion) # Entropy measures the uncertainty in a specific distribution
    return entropy

def cond_entropy(dataframe, attribute, target_attribute):
    '''Calculates the conditional entropy from the dataset and the given attribute'''
    #attribute outlook, temp, humidity...
    attribute = str(attribute)
    attribute_data = dataframe[attribute]
    labels = attribute_data.unique() #Play has two options 'Yes', 'No'
    targets = dataframe[target_attribute].unique()
    entropy_attribute = 0
    proportion_list = []
    
    for label in labels:
        entropy = 0  #Initialize Entropy
        for target in targets:
            num = len(dataframe[dataframe[attribute] == label][dataframe[target_attribute] ==target]) #Number of times the attribute has label and has target 
            denom = len(dataframe[dataframe[attribute]==label]) +eps #denominator
            proportion = num/denom #Proportion of given value in the dataset
            entropy += -proportion*log(proportion+eps) # Entropy measures the uncertainty in a specific distribution
        
        proportion_attribute = denom/len(dataframe[attribute])
        proportion_list.append(proportion_attribute)
        
        entropy_attribute += -proportion_attribute*entropy
    return abs(entropy_attribute), proportion_list


def information_gain(Entropy_parent, Entropy_child):
    '''Calculates the information gain from the parent and child entropy'''
    return (Entropy_parent- Entropy_child)
    
def gain_ratio(gain, proportion_list):
    '''Calculates the gain ratio for the information gain and proportion list of the attributes'''
    split_entropy = 0
    for prob in proportion_list:
        if prob != 0:
            split_entropy += (prob * log(prob))
    split_entropy *= -1
    if split_entropy == 0:
        return split_entropy
    else:
        return gain / split_entropy
            
        
    
def find_winner(dataframe, Attribute_list, target_attribute, Entropy_parent, criteria):
    '''Determines the best attribute from the spliting criteria of the dataframe pass thhrough'''
    master_gain_list = []
    
    for attribute in Attribute_list:
        child_entropy, prop_list = cond_entropy(dataframe,attribute, target_attribute)
        
        if criteria == 'Information Gain':
            IG = information_gain(Entropy_parent, child_entropy)
            master_gain_list.append(IG)
        
        elif criteria == "Gain Ratio":
            gain = information_gain(Entropy_parent, child_entropy)
            GR = gain_ratio(gain, prop_list)
            master_gain_list.append(GR)

    return Attribute_list[np.argmax(master_gain_list)] 


def get_node(node_dataframe,Attribute_list, criteria):
    '''Determines the best attribute to be the node of the tree for the splitting criteria and the dataframe of the node'''
    #dataframe of node, 
    parent_entropy = entropy(node_dataframe, class_attribute)
    parent_node = find_winner(node_dataframe, Attribute_list, class_attribute, parent_entropy, criteria)
    
    node_values = node_dataframe[parent_node].unique()
    
    return parent_node, node_values
    


def print_conf_mat(y_true, y_predicted):
    '''Prints the confusion matrix from the true and predicted class labels'''
    mat1 = confusion_matrix(y_true, y_predicted)#, labels=["positive", "negative"])
    
    #plt.figure(0)
    ax= plt.subplot()
    
    sns.heatmap(mat1, square=True, annot=True, cbar=False,fmt="d", ax = ax)
    ax.set_title('Predicted Matrix')
    ax.set_xlabel('predicted value')
    ax.set_ylabel('true value')
    plt.show()
    return

def stats_info(list_accuracies):
    '''Detemines the statistical quantities of mean, varience, and std from the list of accuracies'''
    mean = sum(list_accuracies) / len(list_accuracies)
    variance = sum([((x - mean) ** 2) for x in list_accuracies]) / len(list_accuracies)
    std = variance ** 0.5
    print('Mean Cross-Validation Accuracy: \t\t\t{:.2f}'.format(mean))
    print('Standard Deviation of Cross-Validation Accuracy: \t{:.2f}'.format(std))
    print('Variance of Cross-Validation Accuracy: \t\t\t{:.2f}'.format(variance))

    return mean, variance, std

def tree_iteration(dataframe, node_seed, node_subsections, Attribute_list,criteria, tree=None, data_dictionary = None):
    '''Builds a dictionary that represents the decision tree using the ID3 algorithm'''
    
    if tree is None:
        tree = {}
        tree[node_seed] = {}
    #print(Attribute_list)
    if data_dictionary is None:
        data_dict = {}
        data_dict[node_seed] = {}
        
    for  value in node_subsections:
        sub_dataframe = dataframe[dataframe[str(node_seed)] == str(value)] # dataframe of child, parent datafrane with the values 
        data_dict[node_seed][value] = {'data':sub_dataframe} #Stores child dataframe in dictionary
        
        class_labels = sub_dataframe[class_attribute].unique()#Determines the number of unique class labels in child dataframe

        #Base Case, leaf is found
        if (len(class_labels)) <= 1:
            tree[node_seed][value] = sub_dataframe[class_attribute].unique()[0]
            data_dict[node_seed][value] = {'data':sub_dataframe}
            
            
        #else:
        elif (len(class_labels)) > 1:

            data = data_dict[node_seed][value]['data']
            node2, node_values2 = get_node(data, Attribute_list, criteria)
            #Attribute_list.remove(node2)
            tree[node_seed][value], data_dict[node_seed][value]['data']  = tree_iteration(data, node2, node_values2,Attribute_list,criteria)
            
    return tree, data_dict

def train(dataframe,criteria):
    '''Trains the classifier by building the decision tree in a dictionary and returns the main node the data was split by and the dicision tree'''
    
    #Get the list of attribute names from the dataframe
    Data_Attribute_list = list(dataframe.columns.values)# Makes a list of all the column names
    Data_Attribute_list.remove(class_attribute) #Remoces the class attribute from the list of column names
        
    #Determine the main node to split the data using the criterua
    main_node, main_node_values = get_node(dataframe,Data_Attribute_list,criteria)

    #Build the tree using the ID3 algorithm
    tree, data = tree_iteration(dataframe, main_node, main_node_values, Data_Attribute_list, criteria, tree = None ,data_dictionary = None)
    
    return main_node, tree, data

def branch_recursion(dictionary,main_node,Row, empty_list):
    '''Given the row of from the testing dataframe and the decision tree dictionary, this function returns the predicted class label'''
    
    feature_value = Row[main_node].values[0]
    
    for key, val in dictionary.items():

        #Case - The training data did not have sample in test dataframe -- sends training data assuming the first label in keys
        if feature_value not in dictionary.keys():
            feature_value = list(dictionary.keys())[0]
            
        #Case - The feature value in the dataset is equal to the key in the decision tree
        if feature_value == key:
            
            #If the value in the decision tree dictionary is a str, then its the class label and it assigns that label
            if isinstance(val, str):
                #print('Leaf') 
                empty_list.append(val)
                
            #If feature value not at a leaf, follows new branch  
            elif isinstance(val, dict):
                #print('branch')
                for key1, val1 in val.items():
                    sub_dict = val1.get(key1)
                    sub_node = key1
                    sub_dict = val1

                    #Recursion through the function until a leaf is found
                    branch_recursion(sub_dict,sub_node,Row,empty_list)
            
            break # breaks for loop once the feature and key match
        
    return empty_list # return the list with a single value of the predicted class name


def predict(tree_dict, root_node, test_row):
    '''Takes the row in test dataframe and return the predicted value based on the trained decision tree'''
    
    main_dictionary = tree_dict.get(root_node)
    prediction = []
    y_pred1 = branch_recursion(main_dictionary,root_node,test_row, prediction)
    return y_pred1[0]

def fold_cross_val(dataframe, num_folds, criteria):
    '''Cross validation using 10-fold method'''
    
    #Shuffle Dataframe
    df_shuffle = dataframe.iloc[np.random.permutation(len(dataframe))]
    df_shuffle = df_shuffle.reset_index(drop=True) #Reset the index to begin at 0

    folds = num_folds    #Calls number of folds
    fold_size = int(df_shuffle.shape[0]/folds) # Determines the size of the folds
    
    accuracy_list = [] #makes empty list to store accuracy values
    y_pred_master = []
    y_test_master = []
    
    start = 0 # initalize the start
    end = fold_size # initalize the end
    
    for i in range(folds):
        print('\t Calculating fold number {} of {} number if folds...'.format(i+1, folds))

        #For the final cut, if the fold makes a sliver of data left over, the test data will take the extra data. 
        len_dataframe = len(df_shuffle)
        if (len_dataframe - end) < fold_size:
            end = len_dataframe
            
        df_test = df_shuffle.iloc[start:end] #dataframe of test values from the fold
        y_test = df_test.iloc[:,-1] #True values labeled
        df_test = df_test.drop(labels=class_attribute, axis=1) # removes the label column from df_test
        
        #print(df_test)
        
        drop_index = list(range(start,end))
        df_train = df_shuffle.drop(drop_index) #, axis = 0)
        
        #print(df_test)
        #print(y_test)
        start += fold_size
        end += fold_size
    
        main_node, train_tree, train_data = train(df_train, criteria)
        y_pred_list_fold = []
    
        for i in range(len(df_test)):
            Row_i = df_test.iloc[i:i+1]
            y_pred_list_fold.append(predict(train_tree, main_node, Row_i))
        
        y_pred_master.append(y_pred_list_fold)
        y_test_master.append(y_test)
        
        accuracy_fold1 = accuracy(y_test, y_pred_list_fold)
        accuracy_list.append(accuracy_fold1)
    return accuracy_list, y_pred_master, y_test_master

def cross_validation(times,dataframe, criteria):
    '''Runs the cross validation number of times, in this case 10 time'''
    
    master_acc = []
    master_y_pred = []
    master_y_test = []
    
    print('Decision Tree using the criteria of == {}...'.format(criteria))
    for i in range(times):
        print('Calculating {} of {} times - 10 fold cross validation...'.format(i, times))
        accuracy_list, y_pred_master, y_test_master = fold_cross_val(dataframe, num_folds = 10, criteria = criteria)
        
        master_acc.append(accuracy_list)
        master_y_pred.append(y_pred_master)
        master_y_test.append(y_test_master)
        #print(master_acc)    
    return master_acc, master_y_pred, master_y_test

def main():
    
    #Running the cases
    accuracy_IG, y_pred_IG, y_test_IG = cross_validation(10, df, criteria = "Information Gain")
    accuracy_GR, y_pred_GR, y_test_GR = cross_validation(10, df, criteria = "Gain Ratio")
    
    
    #Post Processing
    #Information Gain............................
    
    accuracy_IG_flat = [y for x in accuracy_IG for y in x]
    
    stats_info(accuracy_IG_flat)
    y_pred_IG_flat = [y for x in y_pred_IG for y in x]
    y_pred_IG_flat = [y for x in y_pred_IG_flat for y in x]
    
    
    y_test_IG_flat_sub = [list(y) for x in y_test_IG for y in x]
    y_test_IG_flat = [y for x in y_test_IG_flat_sub for y in x]
    
    
    print_conf_mat(y_test_IG_flat, y_pred_IG_flat)
    
    #Gain Ratio............................
    accuracy_GR = accuracy_GR
    accuracy_GR_flat = [y for x in accuracy_GR for y in x]
    stats_info(accuracy_GR_flat)
    
    y_pred_GR_flat = [y for x in y_pred_GR for y in x]
    y_pred_GR_flat = [y for x in y_pred_GR_flat for y in x]
    
    y_test_GR_flat_sub = [list(y) for x in y_test_GR for y in x]
    y_test_GR_flat = [y for x in y_test_GR_flat_sub for y in x]
    
    print_conf_mat(y_test_GR_flat, y_pred_GR_flat)
    return
 
#%%MAIN........................................................................
main()
    

#%%
    
        