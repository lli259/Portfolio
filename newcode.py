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


TIME_MAX=200
PANELTY_TIME=200
NORMALIZE=1
Medium_diff=0

def getError(y, y_):
	print(str(-relative_score(y, y_))+"  /  "+str(-max_relative_score(y, y_))+"  /  "+str(mean_squared_error(y, y_)**0.5))

def perSolvedandAveTime(p,l):
	print(p,sum([1 if i<TIME_MAX-1 else 0 for i in l])/float(len(l))," / ",sum(l)/float(len(l)))
	#pass

def printportfolio(df3):
	fulltime=TIME_MAX
	fiexedRT=60 # set 100 seconds to run 3 algorithms
	timeForPred=[PANELTY_TIME]*len(df3)
	for i in range(len(timeForPred)):
		if df3[i][0] <= fiexedRT:
			timeForPred[i]=	df3[i][0]
		else:
			if df3[i][1] <= fiexedRT:
				timeForPred[i]=	df3[i][1]+fiexedRT
			else:
				if df3[i][2] <= fiexedRT:
					timeForPred[i]=	df3[i][2]+2*fiexedRT

				else:
					if df3[i][0]<=fulltime-2*fiexedRT:
						timeForPred[i]=	df3[i][0]+2*fiexedRT

	perSolvedandAveTime("portfolio",thirdruntime)


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

score_functions=[make_scorer(relative_score),make_scorer(max_relative_score),"neg_mean_squared_error"]
score_f=score_functions[2]

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
featureList=allCombine.columns.values[:-len(algorithmNames)]

print(featureList)
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
oracle_value=np.amin(allRuntime.values, axis=1)
oracle_index=np.argmin(allRuntime.values, axis=1)
Oracle_name=[allRuntime.columns[oracle_index[i]].split("_")[1]  for i in range(len(oracle_index))]


allCombine["Oracle_value"]=oracle_value
allCombine["Oracle_name"]=Oracle_name

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


trainResult=trainSet.copy()
validResult=validSet.copy()
testResult=testSet.copy()

for alg in algorithmNames:
	trainSet_X=trainSet.ix[:,featureList].values
	trainSet_y=trainSet["runtime_"+alg].values
	validSet_X=validSet.ix[:,featureList].values
	validSet_y=validSet["runtime_"+alg].values
	testSet_X=testSet.ix[:,featureList].values
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
		kNN_scores = []
		for k in max_neigh:
			kNeigh =KNeighborsRegressor(n_neighbors=k)
			loss = -cross_val_score(kNeigh,trainSet_X, trainSet_y, cv=10, scoring=score_f)
			kNN_scores.append(loss.mean())
		#print "kNNscoring:",kNN_scores
		plt.plot(max_neigh, kNN_scores,label="kNN")
		plt.xlabel('Value of depth: regression_'+alg)
		plt.ylabel('Cross-Validated MSE')
		#plt.show()
		plt.legend()
		bestscoreRF,bestKNeib=sorted(list(zip(kNN_scores,max_neigh)))[0]

		plt.savefig("regression_"+alg)
		plt.clf()

		##print "bestscoreRF:",bestscoreRF


		bestDepth[alg]=(bestDepthDT,bestDepthRF,bestKNeib)
		with open('regression_bestDepth.pickle', 'wb') as handle:
			pickle.dump(bestDepth, handle)

	dtModel=tree.DecisionTreeRegressor(max_depth=bestDepthDT)
	dtModel= dtModel.fit(trainSet_X, trainSet_y)

	y_=dtModel.predict(trainSet_X)
	trainResult["DT_"+alg+"_pred"]=y_
	y_=dtModel.predict(validSet_X)
	validResult["DT_"+alg+"_pred"]=y_
	y_=dtModel.predict(testSet_X)
	testResult["DT_"+alg+"_pred"]=y_

	##########
	rfModel=RandomForestRegressor(max_depth=bestDepthRF)
	rfModel= rfModel.fit(trainSet_X, trainSet_y)
	y_=rfModel.predict(trainSet_X)
	trainResult["RF_"+alg+"_pred"]=y_
	y_=rfModel.predict(validSet_X)
	validResult["RF_"+alg+"_pred"]=y_
	y_=rfModel.predict(testSet_X)
	testResult["RF_"+alg+"_pred"]=y_

    #########
	kNeigh =KNeighborsRegressor(n_neighbors=bestKNeib)
	kNeigh= kNeigh.fit(trainSet_X, trainSet_y)
	y_=kNeigh.predict(trainSet_X)
	trainResult["kNN_"+alg+"_pred"]=y_
	y_=kNeigh.predict(validSet_X)
	validResult["kNN_"+alg+"_pred"]=y_
	y_=kNeigh.predict(testSet_X)
	testResult["kNN_"+alg+"_pred"]=y_


trainResult.to_csv("training_result.csv")
validResult.to_csv("validation_result.csv")
testResult.to_csv("testing_result.csv")






#analysis
#per algorithm:
##solved percent
##time
##dt error on it
##rf
##kNN

runtimeIndex=[i for i in trainResult.columns if "runtime" in i]


#compare Oracle with
#DT
#RF
#kNN
#porfolio predicted values

print("trainSet")
for alg in runtimeIndex:
	perSolvedandAveTime(alg+"",trainResult[alg])
print("oracle_portfolio")
perSolvedandAveTime("",trainResult.Oracle_value.values)


for mName in "DT,RF,kNN".split(","):
	print(mName)
	runtimeIndex=[i for i in trainResult.columns if "runtime" in i]
	kNNsIndex=[i for i in trainResult.columns if mName in i]
	kNNs=trainResult[runtimeIndex+kNNsIndex].copy()
	kNNDf=kNNs[kNNsIndex].copy()
	for i in kNNDf.columns.values:
		kNNDf[i]=[(j,i)for j in kNNDf[i]]
	kNNlist=kNNDf.values
	kNNlist.sort()

	#bestIndex=np.argmin(kNNDf.values, axis=1)
	#bestpredname=[kNNDf.columns[bestIndex[i]]  for i in range(len(bestIndex))]
	#bestname=["runtime_"+i.split("_")[1] for i in bestpredname]

	bestpredname=[i[0][1] for i in kNNlist]
	bestname=["runtime_"+i.split("_")[1] for i in bestpredname]
	bestruntime=[kNNs.ix[i,bestname[i]]  for i in range(len(kNNs))]
	kNNs["1st_ham"]=bestname
	kNNs["1st_time"]=bestruntime
	#getError(y, y_):
	perSolvedandAveTime("1st",bestruntime)

	secondpredname=[i[1][1] for i in kNNlist]
	secondname=["runtime_"+i.split("_")[1] for i in secondpredname]
	secondruntime=[kNNs.ix[i,secondname[i]]  for i in range(len(kNNs))]
	kNNs["2nd_ham"]=secondname
	kNNs["2nd_time"]=secondruntime
	perSolvedandAveTime("2nd",secondruntime)

	thirdpredname=[i[2][1] for i in kNNlist]
	thirdname=["runtime_"+i.split("_")[1] for i in thirdpredname]
	thirdruntime=[kNNs.ix[i,thirdname[i]]  for i in range(len(kNNs))]
	kNNs["3rd_ham"]=thirdname
	kNNs["3rd_time"]=thirdruntime
	perSolvedandAveTime("3rd",thirdruntime)


	accuracyDF=kNNs.loc[:,["1st_ham","2nd_ham","3rd_ham"]].copy()
	#trainSet,validSet
	Oracle_names=trainSet.Oracle_name.copy()
	accuracyDF=accuracyDF.join(Oracle_names)
	oredTop3=accuracyDF.loc[:,["1st_ham","2nd_ham","3rd_ham"]].values
	Oracle_names=accuracyDF.Oracle_name.values
	accResult=[1 if "runtime_"+Oracle_names[i] in oredTop3[i] else 0 for i in range(len(oredTop3))]
	print("Accuracy",sum(accResult)/len(accResult))

	printportfolio(kNNs.loc[:,["1st_time","2nd_time","3rd_time"]].values)
	kNNs.to_csv(("training_result_analysis_"+mName+".csv"))


print("validSet")
for alg in runtimeIndex:
	perSolvedandAveTime(alg+"",validResult[alg])
print("oracle_portfolio")
perSolvedandAveTime("",validResult.Oracle_value.values)

for mName in "DT,RF,kNN".split(","):
	print(mName)
	runtimeIndex=[i for i in validResult.columns if "runtime" in i]
	kNNsIndex=[i for i in validResult.columns if mName in i]
	kNNs=validResult[runtimeIndex+kNNsIndex].copy()
	kNNDf=kNNs[kNNsIndex].copy()
	for i in kNNDf.columns.values:
		kNNDf[i]=[(j,i)for j in kNNDf[i]]
	kNNlist=kNNDf.values
	kNNlist.sort()

	#bestIndex=np.argmin(kNNDf.values, axis=1)
	#bestpredname=[kNNDf.columns[bestIndex[i]]  for i in range(len(bestIndex))]
	#bestname=["runtime_"+i.split("_")[1] for i in bestpredname]

	bestpredname=[i[0][1] for i in kNNlist]
	bestname=["runtime_"+i.split("_")[1] for i in bestpredname]
	bestruntime=[kNNs.ix[i,bestname[i]]  for i in range(len(kNNs))]
	kNNs["1st_ham"]=bestname
	kNNs["1st_time"]=bestruntime
	#getError(y, y_):
	perSolvedandAveTime("1st",bestruntime)

	secondpredname=[i[1][1] for i in kNNlist]
	secondname=["runtime_"+i.split("_")[1] for i in secondpredname]
	secondruntime=[kNNs.ix[i,secondname[i]]  for i in range(len(kNNs))]
	kNNs["2nd_ham"]=secondname
	kNNs["2nd_time"]=secondruntime
	perSolvedandAveTime("2nd",secondruntime)

	thirdpredname=[i[2][1] for i in kNNlist]
	thirdname=["runtime_"+i.split("_")[1] for i in thirdpredname]
	thirdruntime=[kNNs.ix[i,thirdname[i]]  for i in range(len(kNNs))]
	kNNs["3rd_ham"]=thirdname
	kNNs["3rd_time"]=thirdruntime
	perSolvedandAveTime("3rd",thirdruntime)

	accuracyDF=kNNs.loc[:,["1st_ham","2nd_ham","3rd_ham"]].copy()
	#trainSet,validSet
	Oracle_names=validSet.Oracle_name.copy()
	accuracyDF=accuracyDF.join(Oracle_names)
	oredTop3=accuracyDF.loc[:,["1st_ham","2nd_ham","3rd_ham"]].values
	Oracle_names=accuracyDF.Oracle_name.values
	accResult=[1 if "runtime_"+Oracle_names[i] in oredTop3[i] else 0 for i in range(len(oredTop3))]
	print("Accuracy",sum(accResult)/len(accResult))

	printportfolio(kNNs.loc[:,["1st_time","2nd_time","3rd_time"]].values)
	kNNs.to_csv(("validition_result_analysis_"+mName+".csv"))

print("testSet")
for alg in runtimeIndex:
	perSolvedandAveTime(alg+"",testResult[alg])
print("oracle_portfolio")
perSolvedandAveTime("",testResult.Oracle_value.values)

for mName in "DT,RF,kNN".split(","):
	print(mName)
	runtimeIndex=[i for i in testResult.columns if "runtime" in i]
	kNNsIndex=[i for i in testResult.columns if mName in i]
	kNNs=testResult[runtimeIndex+kNNsIndex].copy()
	kNNDf=kNNs[kNNsIndex].copy()
	for i in kNNDf.columns.values:
		kNNDf[i]=[(j,i)for j in kNNDf[i]]
	kNNlist=kNNDf.values
	kNNlist.sort()

	#bestIndex=np.argmin(kNNDf.values, axis=1)
	#bestpredname=[kNNDf.columns[bestIndex[i]]  for i in range(len(bestIndex))]
	#bestname=["runtime_"+i.split("_")[1] for i in bestpredname]

	bestpredname=[i[0][1] for i in kNNlist]
	bestname=["runtime_"+i.split("_")[1] for i in bestpredname]
	bestruntime=[kNNs.ix[i,bestname[i]]  for i in range(len(kNNs))]
	kNNs["1st_ham"]=bestname
	kNNs["1st_time"]=bestruntime
	#getError(y, y_):
	perSolvedandAveTime("1st",bestruntime)

	secondpredname=[i[1][1] for i in kNNlist]
	secondname=["runtime_"+i.split("_")[1] for i in secondpredname]
	secondruntime=[kNNs.ix[i,secondname[i]]  for i in range(len(kNNs))]
	kNNs["2nd_ham"]=secondname
	kNNs["2nd_time"]=secondruntime
	perSolvedandAveTime("2nd",secondruntime)

	thirdpredname=[i[2][1] for i in kNNlist]
	thirdname=["runtime_"+i.split("_")[1] for i in thirdpredname]
	thirdruntime=[kNNs.ix[i,thirdname[i]]  for i in range(len(kNNs))]
	kNNs["3rd_ham"]=thirdname
	kNNs["3rd_time"]=thirdruntime
	perSolvedandAveTime("3rd",thirdruntime)

	accuracyDF=kNNs.loc[:,["1st_ham","2nd_ham","3rd_ham"]].copy()
	#trainSet,validSet
	Oracle_names=testSet.Oracle_name.copy()
	accuracyDF=accuracyDF.join(Oracle_names)
	oredTop3=accuracyDF.loc[:,["1st_ham","2nd_ham","3rd_ham"]].values
	Oracle_names=accuracyDF.Oracle_name.values
	accResult=[1 if "runtime_"+Oracle_names[i] in oredTop3[i] else 0 for i in range(len(oredTop3))]
	print("Accuracy",sum(accResult)/len(accResult))
	printportfolio(kNNs.loc[:,["1st_time","2nd_time","3rd_time"]].values)
	kNNs.to_csv(("testing_result_analysis_"+mName+".csv"))
