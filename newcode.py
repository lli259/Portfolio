import os
import sys
import math
import pandas as pd
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer


plt.switch_backend('agg')

bins=int(sys.argv[1])




def relative_score(y_true, y_pred):

		res=[]
		for i in range(len(y_true)):
			if y_true[i]>y_pred[i]:
				res.append((y_true[i]-y_pred[i])/(y_true[i]))
			else:
				res.append((y_pred[i]-y_true[i])/(y_true[i]))
		return -sum(res)/float(len(res))

def max_relative_score(y_true, y_pred):

		res=[]
		for i in range(len(y_true)):
			if y_true[i]>y_pred[i]:
				res.append((y_true[i]-y_pred[i])/(y_true[i]))
			else:
				res.append((y_pred[i]-y_true[i])/(y_true[i]))
		return -max(res)
def min_relative_score(y_true, y_pred):

		res=[]
		for i in range(len(y_true)):
			if y_true[i]>y_pred[i]:
				res.append((y_true[i]-y_pred[i])/(y_true[i]))
			else:
				res.append((y_pred[i]-y_true[i])/(y_true[i]))
		return -min(res)


#get training data, testing data, validation data
#combine all the featurs + runtime_
def testbins(X,i,binNum):
	bin_size=int(math.ceil(len(X)/binNum))

	if i==0:
		return np.array(X[bin_size:]),np.array(X[:bin_size])
	elif i==4:
		return np.array(X[:(binNum-1)*bin_size]),np.array(X[-bin_size:])
	else:
		return np.append(X[:bin_size*(i)],X[bin_size*(i+1):],axis=0),np.array(X[bin_size*(i):bin_size*(i+1)])


score_functions=[make_scorer(relative_score),make_scorer(max_relative_score),"neg_mean_squared_error"]
score_f=score_functions[2]

TIME_MAX=200
PANELTY_TIME=200
NORMALIZE=1
Medium_diff=0


featureFile="feature_values.csv"
featureValue=pd.read_csv('./csv/'+featureFile)
featureValue=featureValue.set_index("instance_id")
allCombine=featureValue.copy()

algoRTFile="algorithm_runs.csv"
pd1=pd.read_csv('./csv/'+algoRTFile)

algorithmNames=list(set(pd1["algorithm"].values))
algorithmNames=sorted(algorithmNames)

for algo in algorithmNames:
    singleAlg=pd1[pd1["algorithm"]==algo]
    #change index to instance_id
    singleAlg=singleAlg.set_index("instance_id")
    #only save instance_id and running time
    singleAlg=singleAlg[["runtime"]]
    #change "runtime" to "runtime_index" to distinguish different algrithms
    singleAlg.columns=["runtime_"+algo]
    #save the mapping of index to algorithm name
    #print "Instance runtime shape for each algorithm:",algo,singleAlg.shape
    allCombine=allCombine.join(singleAlg)
    allCombine = allCombine[~allCombine.index.duplicated(keep='first')]

allCombine.sort_index()

#print(allCombine.shape)
#print(allCombine.head(2))
featureList=allCombine.columns.values

#drop "na" rows
allCombine=allCombine.dropna(axis=0, how='any')

#drop "?" rows
for feature in featureList[1:]:
	if allCombine[feature].dtypes=="object":
		# delete from the pd1 rows that contain "?"
		allCombine=allCombine[allCombine[feature].astype("str")!="?"]





algs=["runtime_"+algo for algo in algorithmNames]
allRuntime=allCombine[algs]
#print(allRuntime)
allCombine["Oracle"]=np.amin(allRuntime, axis=1)
#allCombine["Oracle"]=
allCombine.sort_values(['num_of_nodes', 'num_of_edges',"bi_edge"], ascending=[True, True,True])

#print(allCombine.shape)
#print(allCombine.head(2))

# get testing data 20% of the full data:
random.seed(1)
testIndex=random.sample(range(allCombine.shape[0]), int(allCombine.shape[0]*0.2))

trainIndex=list(range(allCombine.shape[0]))
for i in testIndex:
	if i in trainIndex:
		trainIndex.remove(i)

testSet=allCombine.iloc[testIndex]
trainSetAll=allCombine.iloc[trainIndex]

trainSet,validSet=testbins(trainSetAll,bins,5)
trainSet=pd.DataFrame(trainSet,columns=trainSetAll.columns)
validSet=pd.DataFrame(validSet,columns=trainSetAll.columns)
print("ALL:",allCombine.shape)
print("trainAll:",trainSetAll.shape)
print("trainSet:",trainSet.shape)
print("validSet:",validSet.shape)
print("testSet:",testSet.shape)

trainSet.to_csv("trainSet.csv",index=False)
validSet.to_csv("validSet.csv",index=False)
testSet.to_csv("testSet.csv",index=False)


#each data
#instanceFeature=allCombine.columns.values[:-(len(algorithmNames)+1)]
#trainSetAll[list(instanceFeature)+["runtime_"+alg]]

#train each model:

bestDepth={}

if os.path.isdir("parameter_pickle"):
    pickleFiles=[pickFile for pickFile in os.listdir('./parameter_pickle') if pickFile.endswith(".pickle")]
    if 'regression_bestDepth.pickle' in pickleFiles:
        with open('./parameter_pickle/regression_bestDepth.pickle', 'rb') as handle:
            bestDepth = pickle.load(handle)

for alg in algorithmNames:
    trainSet_X=trainSet.iloc[:,:-(len(algorithmNames)+1)].values
    trainSet_y=trainSet["runtime_"+alg].values
    validSet_X=validSet.iloc[:,:-(len(algorithmNames)+1)].values
    validSet_y=validSet["runtime_"+alg].values
    testSet_X=testSet.iloc[:,:-(len(algorithmNames)+1)].values
    testSet_y=testSet["runtime_"+alg].values
    bestDepthDT=0
    bestDepthRF=0
    bestKNeib=0
    '''
    scaler = StandardScaler()
	scaler.fit(trainSet_X)

	if NORMALIZE==1:
		trainSet_X=scaler.transform(trainSet_X)
        validSet_X=scaler.transform(validSet_X)
        testSet_X=scaler.transform(testSet_X)
    '''
    pickleFiles=[pickFile for pickFile in os.listdir('.') if pickFile.endswith(".pickle")]
    if 'regression_bestDepth.pickle' in pickleFiles:
        with open('regression_bestDepth.pickle', 'rb') as handle:
            bestDepth = pickle.load(handle)
            bestDepthDT,bestDepthRF,bestKNeib=bestDepth.get(alg,(0,0,0))
    if bestKNeib==0 and bestDepthDT==0 and bestDepthRF==0:

        #Load parameter from pickle
        max_depth = range(2, 30, 1)
        dt_scores = []
        for k in max_depth:
            regr_k =tree.DecisionTreeRegressor(max_depth=k)
            loss = -cross_val_score(regr_k, trainSet_X, trainSet_y, cv=10, scoring=score_f)
            dt_scores.append(loss.mean())
        #print "DTscoring:",dt_scores
        plt.plot(max_depth, dt_scores,label="DT")
        plt.xlabel('Value of depth: Algorithm'+alg)
        plt.ylabel('Cross-Validated MSE')
        #plt.show()
        bestscoreDT,bestDepthDT=sorted(list(zip(dt_scores,max_depth)))[0]
        ##print "bestscoreDT:",bestscoreDT


        max_depth = range(2, 30, 1)
        dt_scores = []
        for k in max_depth:
            regr_k = RandomForestRegressor(max_depth=k)
            loss = -cross_val_score(regr_k, trainSet_X, trainSet_y, cv=10, scoring=score_f)
            dt_scores.append(loss.mean())
        plt.plot(max_depth, dt_scores,label="RF")
        #print "RFscoring:",dt_scores
        plt.xlabel('Value of depth: Algorithm'+alg)
        plt.ylabel('Cross-Validated MSE')
        #plt.show()
        bestscoreRF,bestDepthRF=sorted(list(zip(dt_scores,max_depth)))[0]
        ##print "bestscoreRF:",bestscoreRF

        max_neigh = range(2, 30, 1)
        knn_scores = []
        for k in max_neigh:
            kNeigh =KNeighborsRegressor(n_neighbors=k)
            loss = -cross_val_score(kNeigh,trainSet_X, trainSet_y, cv=10, scoring=score_f)
            knn_scores.append(loss.mean())
        #print "knnscoring:",knn_scores
        plt.plot(max_neigh, knn_scores,label="KNN")
        plt.xlabel('Value of depth: regression_'+alg)
        plt.ylabel('Cross-Validated MSE')
        #plt.show()
        plt.legend()
        bestscoreRF,bestKNeib=sorted(list(zip(knn_scores,max_neigh)))[0]

        plt.savefig("regression_"+alg)
        plt.clf()

        ##print "bestscoreRF:",bestscoreRF


        bestDepth[alg]=(bestDepthDT,bestDepthRF,bestKNeib)
        with open('regression_bestDepth.pickle', 'wb') as handle:
            pickle.dump(bestDepth, handle)


def perSolved(l):
    return sum([1 if i<TIME_MAX-1 else 0 for i in l])/float(len(l))
def averTime(l):
    return sum(l)/float(len(l))
