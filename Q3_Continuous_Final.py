#SYDE 675
#Matthew Cann
#20863891
#Assignment 2 Question 3 - Wine Dataset

#IMPORTS.......................................................................
import pandas as pd
from numpy import log2 as log
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os # accessing directory structure
import numpy as np
import matplotlib.pyplot as plt
import random

eps = np.finfo(float).eps #Small value such that log won't fail

Directory = "E:\Documents\Waterloo-Masters\SYDE 675\Assignment2"

os.chdir(Directory) 



#%%#%% DATASETS...................................................................

def get_wine_dataset():
    df = pd.read_csv('wine.data', delimiter=',')
    df.columns=['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids','Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity','Hue' , 'OD280/OD315 of diluted wines','Proline' ]
    #df.to_excel("outputttt.xlsx")  
    return df

#%% LOAD DATASET...............................................................


df = get_wine_dataset()
class_attribute  = 'Class'
class_column = df[class_attribute]

df.drop(labels=[class_attribute], axis=1,inplace = True) #Removes the column with class labels
df.insert(len(df.columns), class_attribute, class_column) # inserts class label column at the end of the dataframe
#df.to_excel("test_cont.xlsx")  

print(df)

    
#%%FUNCTIONS...................................................................

def accuracy(y_true, y_predicted):
    '''Reports the accuracy of two lists of true and predicted values'''
    correct = 0
    count = 0
    for true, pred in zip(y_true, y_predicted):
        if int(true) == int(pred):
            correct += 1
        else:
            print(count)
        count += 1

    accuracy = correct/len(y_predicted)*100
    print('Accuracy of classifer {:0.2f} %' .format(accuracy))
    return accuracy


def print_conf_mat(y_true, y_predicted):
    '''Prints the confusion matrix from the true and predicted class labels'''

    mat1 = confusion_matrix(y_true, y_predicted) #labels=["positive", "negative"])
    #true_mat = confusion_matrix(y_true,y_true,labels=["positive", "negative"])
    
    #plt.figure(0)
    ax= plt.subplot()
    
    sns.heatmap(mat1, square=True, annot=True, cbar=False,fmt="d", ax = ax)
    ax.set_title('Predicted Matrix')
    ax.set_xlabel('predicted value')
    ax.set_ylabel('true value')
    plt.show()
    
    
    return

def entropy(dataframe, target_attribute):
    '''Calculates the Entropy of a dataset for the target attribute'''
    entropy = 0  #Initialize Entropy
    values = dataframe[target_attribute].unique() #Play has two options 'Yes', 'No'
    
    play_data = list(dataframe[target_attribute].values)

    for value in values:
        proportion = play_data.count(value)/len(play_data) #Proportion of given value in the dataset
        entropy += -proportion*log(proportion) # Entropy measures the uncertainty in a specific distribution
    return entropy



def information_gain(Entropy_parent, Entropy_child):
        return (Entropy_parent- Entropy_child)
    
    
    






#%%
def binary_split_dataframes(threshold, dataframe, node_seed):
    #print('IM IN THE BINARY_SPLIT_DATAFRAME FUCNTION')
    l = list(dataframe.columns) # A list of the names of the dataframe columns
    left_array = np.array(l) # makes an array with the header of dataframe
    right_array = np.array(l)# makes an array with the header of dataframe
    
    for index, row in dataframe.iterrows():
        
        if row[node_seed] < threshold:
            row_array = np.array(row)
            right_array = np.vstack((right_array, row_array)) 
            
        else:
        #if row[node_seed] > threshold:
            row_array = np.array(row)
            left_array = np.vstack((left_array, row_array))
    
    #Edge case, if the value is the min or max of the dataframe[node_seed] then
    #one of the split dataframe will have all the data and the other none. This checks the length of the split dataframes and returns an empty dataframe
    #print(left_array)
    #print(left_array.shape)
    #print(right_array.shape)
    #print(right_array)
    
    if len(left_array.shape) == 1 or len(right_array.shape) == 1: #This truth statement says if the shape of the array only has one entry, then it has the column titles and no entries and thus is empty. 
        left_df = pd.DataFrame()
        right_df = pd.DataFrame()
        
    else:
        #if left_array == 
        l_columns= left_array[0]
        l_values = left_array[1:,]
        r_columns= right_array[0]
        r_values = right_array[1:,]
    
        left_df = pd.DataFrame(data = l_values, columns = l_columns) 
        right_df = pd.DataFrame(data = r_values, columns = r_columns) 
        
    return left_df, right_df


print(df.shape)
print(df)



#%%

def find_gain_continous(dataframe,child_entropy, target_attribute):
    Entropy_parent = entropy(dataframe, target_attribute)
    IG = information_gain(Entropy_parent, child_entropy)
    return IG

def threshold_index(dataframe, feature_name, ):
    '''This function takes the dataframe and determines the indecese where it is most likely to be a threshold value,
    this occurs at class values that are different than the previous. The function returns the list of indices determined'''
    
    df_sort = dataframe.sort_values(by = feature_name) #Keeps the same indices as the dataframe
    indexes = df_sort.index # Gets indexes of the sorted data

    classes = df_sort[class_attribute] # List of class labels
    
    prev_class = classes[indexes[0]] #Sets the inital prev class as the first class so that it doesn't include it
    
    index_list = []
    
    for index, class_label in zip(indexes, classes): #Since shuffled df has reset axis this enurmate is ok with constitant index
        if prev_class != class_label:
            index_list.append(index)
        prev_class = class_label
        
    return index_list


def find_cont_split(dataframe, criteria):
    '''Calculates the conditional entropy from the dataset and the given attribute'''

    gain_tmp = 0
    threshold = 0 
    node = 0
    final_split_left = pd.DataFrame()
    final_split_right = pd.DataFrame()
    
    features = dataframe.columns[:-1] # Features of the dataframe minus the class label    
    
    for feature in features:
        
        index_list = threshold_index(dataframe, feature)
        
        #node_values = dataframe[feature]
        #node_values = list(node_values)

        #for index in range(len(node_values)):
        
        for index in index_list:

            #print('Computing for value {} out of {} values'.format(index+1, len(node_values)))
            #value = node_values[index]
            
            value = dataframe[feature][index]
            #value = 0.2 #threshold
            left_df, right_df = binary_split_dataframes(value,dataframe,feature)
            #Edge case if the daatframe is empty it returns a low gain to not be pass through
            if left_df.empty or right_df.empty:
                gain = 0
            
            else:
                gain = split_criteria(dataframe, left_df, right_df, criteria)
                
                
            if gain > gain_tmp:
                node = feature 
                gain_tmp = gain
                threshold = value
                final_split_left = left_df
                final_split_right = right_df
                    
            #print('\t Testing Feature: {}'.format(feature) )   
            #print('\t Current best feature: {}'.format(node))    
            #print('\t Current best splitting value: {}'.format(threshold))
            #print('\t Gain of the current winnner: {}'.format(gain_tmp))
            
            
    return node, threshold, final_split_left, final_split_right



def split_criteria(dataframe, left_df, right_df, criteria):
    '''Calculates either the information gain or gain ratio and returns the value'''

    proportion_list = []
    
    #Determine the entropy of both split datasets    
    entropy_l = entropy(left_df, class_attribute)
    entropy_r = entropy(right_df, class_attribute)
    prop_l = left_df.shape[0]/dataframe.shape[0]
    prop_r = right_df.shape[0]/dataframe.shape[0]

    #Adds proportions to list for gain ratio calculation  
    proportion_list.append(prop_l)
    proportion_list.append(prop_r)
    
    #Condiotnal entropy of the split
    conditional_entropy = entropy_l*prop_l + entropy_r*prop_r
    #Information gain calculated
    gain = find_gain_continous(dataframe,conditional_entropy,class_attribute)
                
    if criteria == 'Information Gain':
        return gain
    
    elif criteria == "Gain Ratio":
        split_entropy = 0
        for prob in proportion_list:
            if prob != 0:
                split_entropy += (prob * log(prob))
        split_entropy *= -1
        if split_entropy == 0:
            return split_entropy
        else:
            return gain / split_entropy

def continous_tree_iteration(node_seed, threshold_value, branch_dataframes, Attribute_list, criteria, tree=None,data_dictionary = None):
    '''Builds a dictionary that represents the decision tree using the ID3 algorithm'''

    #Finds the optimal thershold value and the two dataframes left and right pf branch
    
    if tree is None:
        tree = {}
        tree[node_seed] = {}
    #print(Attribute_list)
    if data_dictionary is None:
        data_dict = {}
        data_dict[node_seed] = {}
        
    tree[node_seed] = {'threshold': threshold_value}
    data_dict[node_seed] = {'threshold': threshold_value}
    
    
    for direction, sub_dataframe in zip(['greater than', 'less than'],branch_dataframes):
        
        data_dict[node_seed][direction] = sub_dataframe #Save the branch dataframe in the dictionary

        class_labels = sub_dataframe[class_attribute].unique()#Determines the number of unique class labels in child dataframe
        
        trial = sub_dataframe.iloc[:, :-1]
        
        count = 0 
        
        if trial[trial.duplicated()].shape[0] / sub_dataframe.shape[0] == 0.5:
            count+=1
            tree[node_seed][direction] = random.choice(class_labels)
            data_dict[node_seed][direction] = sub_dataframe

        if len(class_labels) <= 1:
            count+=1
            tree[node_seed][direction] = class_labels[0]
            data_dict[node_seed][direction] = sub_dataframe
            
        elif len(Attribute_list) == 1:
            count+=1
            mylist = sub_dataframe[class_attribute]
            from collections import Counter
            c = Counter(mylist)
            tree[node_seed][direction] =  c.most_common(1)[0][0]

        elif count == 0:
            data = data_dict[node_seed][direction]
            
            sub_node, sub_threshold, sub_dataframe_left, sub_dataframe_right =  find_cont_split(data,criteria)

            sub_branches =  [sub_dataframe_left, sub_dataframe_right]
            
            tree[node_seed][direction] , data_dict[node_seed][direction] = continous_tree_iteration(sub_node, sub_threshold, sub_branches, Attribute_list, criteria)

    return tree, data_dict


def train_cont(dataframe, criteria):
    '''Trains the classifier by building the decision tree in a dictionary and returns the main node the data was split by and the dicision tree'''

    #Error message if the incorrect criteria is inputted
    if criteria != 'Information Gain' and criteria != "Gain Ratio":
        print('Decision Tree Error: Pick either Information Gain or Gain Ratio')
        
    Data_Attribute_list = list(dataframe.columns.values)# Makes a list of all the column names
    Data_Attribute_list.remove(class_attribute) #Remoces the class attribute from the list of column names
    
    main_node, main_threshold, main_dataframe_left, main_dataframe_right =  find_cont_split(dataframe, criteria)
    main_branches_dataset = [main_dataframe_left, main_dataframe_right]

    tree, tree_data = continous_tree_iteration(main_node, main_threshold, main_branches_dataset, Data_Attribute_list, criteria,  tree=None,data_dictionary = None)
    return main_node, tree, tree_data



def branch_recursion_continous(dictionary,main_node,Row, empty_list):
    '''Main function in the predict function that takes the row of data in the 
    test dataframe and classifies it from the decision tree dictionary'''
    
    feature_value = Row[main_node].values[0] # Get the value from the Row in main node spot
    threshold_value = dictionary['threshold']
    
    #print(feature_value)
    #print(threshold_value)
    
    if float(feature_value) >= float(threshold_value):
        #print('greater than')
        val = dictionary['greater than']
        #print(val)     
        if isinstance(val, str):
                #print('Leaf') 
                empty_list.append(float(val))
                #print(val)
                
        elif isinstance(val, dict):
            #print(val)
            #print('branch')
            sub_node = list(val.keys())[0]
            sub_dict = val.get(sub_node)
            branch_recursion_continous(sub_dict,sub_node,Row,empty_list)
    else:
        #print('less than')
        val = dictionary['less than']
        #print(val)     
        if isinstance(val, str):
                #print('Leaf') 
                empty_list.append(float(val))
                #print(val)
                
        elif isinstance(val, dict):
            #print(val)
            #print('branch')
            
            sub_node = list(val.keys())[0]
            sub_dict = val.get(sub_node)
            branch_recursion_continous(sub_dict,sub_node,Row,empty_list)
    return empty_list         
            
def predict_continous(tree_dict, test_row):
    '''Given the row of test data, recursively travels down the decision tree and assigns value'''
    node_seed = list(tree_dict.keys())[0]
    main_dictionary = tree_dict.get(node_seed)
    prediction = []
    y_pred1 = branch_recursion_continous(main_dictionary,node_seed,test_row, prediction)
    return y_pred1[0]


def make_dirty(dataframe, percent_dirty ):
    '''Given a dataframe and level of noise - returns a dataframe that has added noise for discrete dataframes'''

    dataframe_dirty = dataframe.copy()
    attributes = list(dataframe.columns) # Gets columns names
    if class_attribute in attributes:
        attributes.remove(class_attribute) # removes the class attraibute name such that the class label does not get changed

    #This loop iterates through the columns and picks randomly L% of rows and changes the values into one of the other attribute values. 
   
    for attribute in attributes: # Iterates through the columns
        column = dataframe[attribute]
        mu = column.mean()
        sigma = column.std()
        
        
        columnupdate = column.sample(frac = percent_dirty)
        for index in list(columnupdate.index):
            value = columnupdate[index]
            s = np.random.normal(mu, sigma, 1)
            new_value = value+s
            columnupdate.at[index] = new_value
        dataframe_dirty[attribute].update(columnupdate)
    return dataframe_dirty

def add_noise(case, X, Y, level):
    '''For the case passed, makes the corresponding noise added dataframe for the passed level and returns the training and testing datasets
    X--  Training dataset
    Y --  Testing dataset'''
    
    if case == "CvC":
        #Training is clean, testing in clean
        return X, Y
    
    if case == "DvC":
        #Training is dirty, testing in clean
        X_dirty = make_dirty(X, level)
        return X_dirty, Y
    
    if case == "CvD":
        #Training is clean, testing in dirty
        Y_dirty = make_dirty(Y, level)
        return X, Y_dirty
    
    if case == "DvD":
        #Training is dirty, testing in dirty
        X_dirty = make_dirty(X, level)
        Y_dirty = make_dirty(Y, level)
        
        return X_dirty, Y_dirty
    
def class_misclassification(dataframe, percent_dirty):
    '''Adds misclassification class label noise to the dataframe with a level passed
    '''
    
    dataframe_mis = dataframe.copy()
    df_update = dataframe_mis.sample(frac = percent_dirty)
    #print(df_update)

    for index in list(df_update.index):
        
        class_label = df_update[class_attribute][index]
        #print(class_label)
        
        class_values = [1, 2, 3] # Possible        
        class_values.remove(class_label)
        new_value = random.choice(class_values)
            
        df_update[class_attribute].at[index] = new_value
    dataframe_mis.update(df_update)
    return dataframe_mis


def class_contradictory(dataframe, percent_dirty):
    '''Takes the dataframe and samples a percentage of the rows and flips the class labels then adds the data to the end of the dataframe'''
    dataframe_cond = dataframe.copy()

    df_update = dataframe_cond.sample(frac = percent_dirty)
    for index in list(df_update.index):
        class_label = df_update[class_attribute][index]
        #print(class_label)
        class_values = [1, 2, 3] # Possible        
        class_values.remove(class_label)
        new_value = random.choice(class_values)

        df_update[class_attribute].at[index] = new_value
        
    dataframe_cond  = dataframe_cond.append(df_update, ignore_index=True)
    return dataframe_cond

def train_test_split(shuffled_df, start_index, end_index, fold_size):
    '''Splits the shuffled dataframe into test, train and y_test for the start and end points'''
    len_dataframe = len(shuffled_df)
    
    if (len_dataframe - end_index) < fold_size:
        end_index = len_dataframe
                
    df_test = shuffled_df.iloc[start_index:end_index] #dataframe of test values from the fold
    y_test = df_test.iloc[:,-1] #True values labeled
    df_test = df_test.drop(labels=class_attribute, axis=1) # removes the label column from df_test


    drop_index = list(range(start_index,end_index))
    df_train = shuffled_df.drop(drop_index) #, axis = 0)
    start_index += fold_size
    end_index += fold_size
        
    return df_test, y_test, df_train, start_index, end_index

def fold_cross_val(dataframe, num_folds, criteria, noise_info):
    '''Cross validation using 10-fold method'''

    case = noise_info[0]
    level = noise_info[1]
    
     #Shuffle Dataframe
    df_shuffle = dataframe.iloc[np.random.permutation(len(dataframe))]
    df_shuffle = df_shuffle.reset_index(drop=True) #Reset the index to begin at 0

    print('\t Decision Tree using the criteria of == {}...'.format(criteria))

    folds = num_folds    #Calls number of folds
    
    fold_size = int(df_shuffle.shape[0]/folds) # Determines the size of the folds
    
    y_pred_master = []
    y_test_master = []
    
    accuracy_list = [] #makes empty list to store accuracy values
    start = 0 # initalize the start
    end = fold_size # initalize the end
    
    for i in range(folds):
        
        print('\t Calculating fold number {} of {} number if folds...'.format(i+1, folds))
        #For the final cut, if the fold makes a sliver of data left over, the test data will take the extra data. 
        
        df_test, y_test, df_train, start, end = train_test_split(df_shuffle,start, end, fold_size)
        
        #Addition of noise to attributes
        #train_noise, test_noise  = add_noise(case, df_train, df_test, level)
        
        #Class Noise Misclassification
        #test_noise = df_test
        #train_noise = class_misclassification(df_train, level)

        #print(df_train)
        #Class noise Contradictory
        test_noise = df_test
        train_noise = class_contradictory(df_train, level)

        main_node, train_tree, train_data = train_cont(train_noise, criteria)
        
        
        y_pred_list_fold = []
        
        for i in range(len(test_noise)):
            Row_i = test_noise.iloc[i:i+1]
            y_pred_list_fold.append(predict_continous(train_tree, Row_i))
            
        y_pred_master.append(y_pred_list_fold)
        y_test_master.append(y_test)
        
        accuracy_fold1 = accuracy(y_test, y_pred_list_fold)
        accuracy_list.append(accuracy_fold1)
        print(accuracy_list)
    return accuracy_list, y_pred_master, y_test_master

def stats_info(list_accuracies):
    
    mean = sum(list_accuracies) / len(list_accuracies)
    variance = sum([((x - mean) ** 2) for x in list_accuracies]) / len(list_accuracies)
    std = variance ** 0.5
    print('Mean Cross-Validation Accuracy: \t\t\t{:.2f}'.format(mean))
    print('Standard Deviation of Cross-Validation Accuracy: \t{:.2f}'.format(std))

    return mean

def cross_validation(times,dataframe, criteria, noise_info):
    master_acc = []
    master_y_pred = []
    master_y_test = []
    print('Decision Tree using the criteria of == {}...'.format(criteria))

    for i in range(times):
        print('Calculating {} of {} times - 10 fold cross validation...'.format(i+1, times))
        
        accuracy_1, y_pred_master, y_test_master = fold_cross_val(dataframe, num_folds = 10, criteria = "Information Gain", noise_info = noise_info)
        
        master_acc.append(accuracy_1)
        master_y_pred.append(y_pred_master)
        master_y_test.append(y_test_master)
        print(master_acc)    
        
    return master_acc, master_y_pred, master_y_test
#%%
def main():
    '''Executes main part of code'''
    
    #Attribute Noise
        
    Attr_noise_level = [0.05, 0.10, 0.15]
    
    cases = ["CvC", "DvC", "CvD", "DvD"]
     
    master_mean_list = []
    
    for case in cases:
        Attr_noise_level = [0.05, 0.10, 0.15]
        mean_list = []
        for level in Attr_noise_level:
            #accuracy_1, y_pred_master, y_test_master = fold_cross_val(df, num_folds = 10, criteria = "Information Gain", noise_info = [case, level])
            
            master_acc, master_y_pred, master_y_test = cross_validation(10, df, criteria = "Information Gain", noise_info = [case, level])
            
            master_acc_flat = [y for x in master_acc for y in x]
            mean = stats_info(master_acc_flat)
            
            mean_list.append(mean)
            print("Final mean list for {}".format(case), mean_list)
        master_mean_list.append(mean_list)
        
    #Class Noise
    
    Class_noise_level = [0.05, 0.10, 0.15]
    mean_list = []
    for level in Class_noise_level:
        #accuracy_1, y_pred_master, y_test_master = fold_cross_val(df, num_folds = 10, criteria = "Information Gain", noise_info = [case, level])
        
        master_acc, master_y_pred, master_y_test = cross_validation(10, df, criteria = "Information Gain", noise_info = [case, level])
        
        master_acc_flat = [y for x in master_acc for y in x]
        mean = stats_info(master_acc_flat)
        
        mean_list.append(mean)
        print("Final mean list for {}".format(case), mean_list)
        
    master_mean_list.append(mean_list)
    
    #HARD CODED RESULTS.............................
    

    CvC = [91.18872549019,89.58333,89.338235]
    DvC = [90.93137255, 89.227941176, 88.180147]  
    CvD = [88.64583333333331, 84.1911764705882, 83.26593137254899]
    DvD = [86.88112745098036, 84.11151960784312, 83.40073529411764]
    Misclas = [82.92892156862744, 75.60661764705881, 70.42892156862743] 
    Condict = [86.67279411764703, 83.70098039215688, 78.16789215686273]
    
    
    plt.figure(0)
    plt.xlabel('Attribute Noise Level')
    plt.ylabel('Accuracy')
    plt.plot(Attr_noise_level, CvC, label = 'CvC')    
    plt.plot(Attr_noise_level, DvC,label = 'DvC')    
    plt.plot(Attr_noise_level, CvD, label = 'CvD')    
    plt.plot(Attr_noise_level, DvD, label = 'DvD') 
    plt.legend()   
    plt.savefig('Wine_Attribute_noise.png')
    plt.show()
    
    
    plt.figure(0)
    plt.xlabel('Class Noise Level')
    plt.ylabel('Accuracy')
    plt.plot(Attr_noise_level, Misclas, label = 'Misclassification')    
    plt.plot(Attr_noise_level, Condict, label  = 'Contradictory')  
    plt.legend()  
    plt.savefig('Wine_class_noise.png')
    
    plt.show()
    
    return
#%%MAIN........................................................................
main()

#%%
   