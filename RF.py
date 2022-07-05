import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings("ignore")


# calculate RF 
#----
def RF_feature(shallow, deep, max_depth, estimators, name, features=False):
    shallow['depth_label'] = np.zeros(shallow.shape[0], dtype=int)
    deep['depth_label'] = np.ones(deep.shape[0], dtype=int)
    
    #select shalow randomly
    shallow_sample = shallow.sample(n=deep.shape[0], replace=False, random_state=42, axis=0)
   
    #merge
    df =pd.concat([shallow_sample, deep])
    
    #train and test
    if features:
        X = df[features].copy()
    else:
        X = df.drop(columns=['apparent_stress', 'depth', 'depth_label', 'Er/Mw']).copy()
    Y = df.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42, stratify=Y['depth_label'])
    clf = RandomForestClassifier(n_estimators= estimators, max_depth=max_depth, class_weight='balanced', random_state=42)
    model = clf.fit(X_train, y_train['depth_label'])
    print('\n')
    print('Training: {}; Testing: {}'.format(X_train.shape[0], X_test.shape[0]))
    print('Classification accuracy {:.3f}'.format(clf.score(X_test, y_test['depth_label'])))
    #============#
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test['depth_label'], y_pred))
    # y_test['pred_label'] = y_pred
    # y_test.to_csv('out/file/'+name+'.csv',index=False,sep=',')
    #==============#
    print('Feature importance:')
    idx = np.argsort(clf.feature_importances_,)
    features_order = []
    feature_importance = []
    feature_acc = [0.5]
    for i in idx[::-1]: 
        print('{:s} {:.3f}'.format(X.columns[i], clf.feature_importances_[i]))
        features_order.append(X.columns[i])
        feature_importance.append(clf.feature_importances_[i])

    for i in range(len(features_order)):
        clf2 = RandomForestClassifier(n_estimators=10, max_depth=3, class_weight='balanced', random_state=12)
        clf2.fit(X_train[features_order[0:i+1]], y_train['depth_label'])
        feature_acc.append(clf2.score(X_test[features_order[0:i+1]], y_test['depth_label']))   
    AccracyFeature(feature_acc, feature_importance, features_order)   

# The changes in classification accuracy by progressively adding the 
# features as input and retraining and re-evaluating the classifier
def AccracyFeature(accuracy, importance, features):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(accuracy)), accuracy, c="#20639B")
    plt.ylabel("Accuracy (%)", fontsize=15)
    plt.xticks([])
    plt.xlim(0, (len(importance)+1))
    plt.yticks([0.4, 0.5, 0.75, 1.0], ['', 50, 75, 100], fontsize=10)
    plt.subplot(2, 1, 2)
    plt.scatter(np.arange(len(importance))+1, importance, c="#ED553B")
    plt.ylabel("Weight", fontsize=15)
    plt.xticks(np.arange(len(importance))+1, features, rotation=45, fontsize=10)
    #plt.axis([0, 0.3, 0, int(len(importance)+1)])
    plt.xlim(0, (len(importance)+1))
    plt.subplots_adjust(hspace=0)
    plt.show()
    

# Repeat 100 times and calculate the mean accuracy and standard error
def RF_score(shallow, deep, max_depth, estimators, features=False):
    score = []
    shallow['depth_label'] = np.zeros(shallow.shape[0], dtype=int)
    deep['depth_label'] = np.ones(deep.shape[0], dtype=int)
    for i in range(100):
        
        # Randomly select some shallow (intermediate-depth) EQs to match the total number of deep (deep-focus)EQs
        shallow_sample = shallow.sample(n=deep.shape[0], replace=False, random_state=i, axis=0)
        
        # merge
        df =pd.concat([shallow_sample, deep])
        
        # train and test
        if features:
            X = df[features].copy()
        else:
            X = df.drop(columns=['apparent_stress', 'depth', 'depth_label', 'Er/Mw']).copy()
        Y = df.copy()
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=i, stratify=Y['depth_label'])
        clf = RandomForestClassifier(n_estimators= estimators, max_depth=max_depth, class_weight='balanced', random_state=i)
        clf.fit(X_train, y_train['depth_label'])
        score.append(clf.score(X_test, y_test['depth_label']))

    print("Average accuracy, std:\n", np.mean(np.asarray(score)), np.std(np.asarray(score)))
#------------#


# dichotomy
#----------#
def linear_score(shallow, deep, features=False):
    shallow['depth_label'] = np.zeros(shallow.shape[0], dtype=int)
    deep['depth_label'] = np.ones(deep.shape[0], dtype=int)
   
    #select
    score = []
    feature_name = []
    
    #  Repeat 100 times and calculate the mean accuracy and standard error
    for i in range(100):
        
        # Randomly select some shallow (intermediate-depth) EQs to match the total number of deep (deep-focus)EQs
        shallow_sample = shallow.sample(n=deep.shape[0], replace=False, random_state=i, axis=0)
        df =pd.concat([shallow_sample, deep])

        #train and test
        if features:
            X = df[features].copy()
        else:
            X = df.drop(columns=['Er/Mw', 'apparent_stress', 'depth', 'depth_label']).copy()
        Y = df['depth_label'].copy()
        
        # mean.std
        accuracy = []
        for feature in X.columns:
            if i == 0:                
                feature_name.append(feature)
            X_singal = X[feature].values
            X_singal = X_singal.reshape(-1, 1)
            clf = LinearSVC(class_weight='balanced')
            clf.fit(X_singal, Y.values.ravel())
            accuracy.append(clf.score(X_singal, Y.values.ravel()))
        score.append(accuracy)
    
    # save the results
    dichotomy = list(zip(feature_name, np.mean(np.asarray(score), axis=0), np.std(np.asarray(score), axis=0)))
    print("dichotomous accuracies:\n Feature\tAcc.\tstd\n", np.array(dichotomy))
    np.savetxt("out/dichotomous_accuracy.txt", np.array(dichotomy), fmt="%s", header="Feature\tAccuracy\tstd")
#---------------------#


# Spearman coefficient
#---------------------#
def spearman_value(df, X, independent):
    spearmans = np.zeros(len(X.columns))
    for i, feature in enumerate(X.columns):
        R = round(spearmanr(df[independent], df[feature])[0], 3)
        spearmans[i] = R
    return spearmans
#---------------------#


def plot_spearman(depth, fm, x_label="fm"):
    plt.figure(1,figsize=(5, 5))
    plt.scatter(fm, depth, c="k")
    plt.xlabel("spearman coefficient of "+ x_label)
    plt.ylabel("spearman coefficient of depth")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
#     plt.savefig("out/spearman.png", format="png", bbox_inches='tight', dpi=600,pad_inches=0.0)
    plt.show()


if __name__ == '__main__':
    # make file
    if not os.path.exists("out/"):
        os.makedirs('out/')
    
    #load data
    df =pd.read_csv('All_data_feature.csv')
    
    # Shallow, intermediate-depth and deep-focus earthquake
    df_shallow =df.loc[df['depth']<=60].copy()
    df_deep =df.loc[(df['depth']>60) & (df['depth']<=700)].copy()
    df_intermediate =df.loc[(df['depth']>60) & (df['depth']<=300)].copy()
    df_deep_focus =df.loc[(df['depth']>300) & (df['depth']<=700)].copy()
    print("The counts of EQS:\nShallow\tdeep\tintermediate-depth\tdeep-focus")
    print(len(df), len(df_shallow), len(df_intermediate), len(df_deep_focus))
    
    features = False

    #------------------------------------------------#
    # Random Forest
    # Shallow vs deep
    print('\nShallow vs deep\n')
    max_depth, n_estimators = 8, 90
    print("depth:", max_depth, "estimator", n_estimators)
    RF_feature(df_shallow, df_deep, max_depth, n_estimators, 'shallow_deep_predict', features=features)
    RF_score(df_shallow, df_deep, max_depth, n_estimators, features=features)
    print('-----------------------------------')
    
    # Intermediate-depth vs deep-focus
    print('\nIntermediate-depth vs deep-focus\n')
    max_depth, n_estimators = 4, 100
    print("depth:", max_depth, "estimator", n_estimators)
    RF_feature(df_intermediate, df_deep_focus, max_depth, n_estimators, 'intermediate_deepfocus_predict', features=features)
    RF_score(df_intermediate, df_deep_focus, max_depth, n_estimators, features=features)
    #------------------------------------------------#

    #--------------------------------------------------#
    # dichotomy
    #deep vs shallow
    print("\tdichotomy")
    print('\nshallow\tvs\tdeep:\n')
    linear_score(df_shallow, df_deep, features=False)
    #intermediate vs deep focus
    print('\nIntermediate-depth\tvs\tdeep-foucus:\n')
    linear_score(df_intermediate, df_deep_focus, features=False)
    #--------------------------------------------------#
    
    #--------------------------------------------------#
    # spearman coefficient
    # All depth
    print("Spearmean coefficient")
    print("All_depth")
    X = df.drop(columns=['depth', 'Er/Mw', 'apparent_stress']).copy()
    # depth, fm, T coefficient
    depth_coef = spearman_value(df, X, "depth")
    fm_coef = spearman_value(df, X, "fm")
    T_coef = spearman_value(df, X, "T_scaled")
    
    # print, save, plot
    spearman_all = list(zip(X.columns, depth_coef, fm_coef, T_coef))
    print("spearman coefficient:\nFeature\tdepth\tfm\tT\n", np.array(spearman_all))
    np.savetxt('out/spearman_shallow_deep', spearman_all, fmt="%s", header="depth\tfm\tT\t")
    plot_spearman(depth_coef, fm_coef)
    
    # > 60 km
    print("\nDepth>60 km")
    df_deep =df.loc[(df['depth']>60) & (df['depth']<=700)].copy()
    X_deep = df_deep.drop(columns=['depth', 'Er/Mw', 'apparent_stress'])
    
    # depth, fm, T coefficient
    deep_depth_coef = spearman_value(df_deep, X_deep, "depth")
    deep_fm_coef = spearman_value(df_deep, X_deep, "fm")
    deep_T_coef = spearman_value(df_deep, X_deep, "T_scaled")
    spearman_all = list(zip(X.columns, deep_depth_coef, deep_fm_coef, deep_T_coef))
    
    #print, save, plot
    print("spearman coefficient:\nFeature\tdepth\tfm\tT\n", np.array(spearman_all))
    np.savetxt('out/spearman_inter_deep.txt', spearman_all, fmt="%s")
    plot_spearman(deep_depth_coef, deep_T_coef, "T_scaled")
    #--------------------------------------------------#