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
inputseed=int(sys.argv[2])

def testbins(X,i):
	bin_size=int(math.ceil(len(X)/5))

	if i==0:
		return np.array(X[bin_size:]),np.array(X[:bin_size])
	elif i==4:
		return np.array(X[:4*bin_size]),np.array(X[-bin_size:])
	else:
		return np.append(X[:bin_size*(i)],X[bin_size*(i+1):],axis=0),np.array(X[bin_size*(i):bin_size*(i+1)])



def write2file(s,b=0):
	with open("allresult.csv","a") as fi:
		fi.write(s)
		if b==1:
			fi.write("\n")
		else:
			fi.write(",")

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

TIME_MAX=200
PANELTY_TIME=200
NORMALIZE=1
Medium_diff=0

score_functions=[make_scorer(relative_score),make_scorer(max_relative_score),"neg_mean_squared_error"]
score_f=score_functions[2]

def getTrainingData():

	#If there are files in training_data and testing_data folders, we skip this.
	#To update information, delete these files or folders.
	if os.path.isdir("regression_training_data") and os.path.isdir("regression_testing_data") and os.path.isdir\
					("classification_training_data") and os.path.isdir("classification_testing_data"):

		reg_trainingCsvFiles=[filename for filename in os.listdir('./regression_training_data') if filename.endswith(".csv")]
		reg_testingCsvFiles=[filename for filename in os.listdir('./regression_testing_data') if filename.endswith(".csv")]
		cls_trainingCsvFiles=[filename for filename in os.listdir('./classification_training_data') if filename.endswith(".csv")]
		cls_testingCsvFiles=[filename for filename in os.listdir('./classification_testing_data') if filename.endswith(".csv")]

		if "algorithm0_training_data.csv" in reg_trainingCsvFiles and "testing_data.csv" in reg_testingCsvFiles \
			and  "training_data.csv" in cls_trainingCsvFiles and "testing_data.csv" in cls_testingCsvFiles:
			#print "Load data directly..."
			return 0

	#print "Preprocess Data..."

	#get instances' running time for each algorithm
	algorithmNames,index2Alg,runTmPerAlgo=getRunTmPerAlgo()

	#get instances' features
	featureValue=getFeatureValue()

	#print "\nBefore preprocess:"
	#print "Feature shape:",featureValue.shape
	#print "Algorithm runtime shape:", runTmPerAlgo[0].shape

	#combine all together, to create <instance, feature, rt1,rt2,rt3.. > for each algorithm
	allCombine=featureValue.copy()
	for algIndex in range(len(algorithmNames)):

		allCombine=allCombine.join(runTmPerAlgo[algIndex])
		#allCombine=pd.concat([allCombine,runTmPerAlgo[algIndex]], axis=1,join_axes=[allCombine.index])
		allCombine = allCombine[~allCombine.index.duplicated(keep='first')]
		#print allCombine.shape

	allCombine.sort_index()
	#print "Feature + Runtime shape:",allCombine.shape

	if not os.path.isdir("combined_notprocessd_data"):
		os.mkdir("combined_notprocessd_data")
	allCombine.to_csv("combined_notprocessd_data/training_data_before_prep.csv")

	#get rid of invalid data
	allValidData=dataRemoveInvalid(allCombine)
	#print "\nFeature + Runtime shape after removing invalid:",allValidData.shape

	allValidData.sort_values(['num_of_nodes', 'num_of_edges',"bi_edge"], ascending=[True, True,True])

	# get testing data 20% of the full data:
	random.seed(inputseed)
	testIndex=random.sample(range(allValidData.shape[0]), int(allValidData.shape[0]*0.2))



	'''
	allValidData["instance_id"]=allValidData.index

	trainSet=pd.DataFrame(columns=allValidData.columns)
	testSet=pd.DataFrame(columns=allValidData.columns)
	for line_index in range(len(allValidData)):
		line=allValidData.iloc[line_index,:]
		##print line
		##print line.instance_id
		if "instrect" in line.instance_id:
			trainSet=trainSet.append(line)
		if "insttri" in line.instance_id:
			testSet=testSet.append(line)

	allValidData=allValidData.set_index("instance_id")
	trainSet=trainSet.set_index("instance_id")
	testSet=testSet.set_index("instance_id")
	'''



	trainIndex=list(range(allValidData.shape[0]))
	for i in testIndex:
		if i in trainIndex:
			trainIndex.remove(i)

	testSet=allValidData.iloc[testIndex]
	trainSet=allValidData.iloc[trainIndex]

	#print "Test Set shape:",testSet.shape
	#print "train Set shape:",trainSet.shape

	#save testing data in file
	if not os.path.isdir("regression_testing_data"):
		os.mkdir("regression_testing_data")
	testSet.to_csv("regression_testing_data/testing_data.csv",index=False)


	instanceFeature=trainSet.columns.values[:-len(algorithmNames)]
	# we get training data for each algorithm
	if not os.path.isdir("regression_training_data"):
		os.mkdir("regression_training_data")


	trainSet1,trainSet2=testbins(trainSet,bins)
	trainSet=pd.DataFrame(trainSet1,columns=trainSet.columns)
	trainSet2=pd.DataFrame(trainSet2,columns=trainSet.columns)

	trainDataProcessed=[]
	for algIndex in range(len(algorithmNames)):
		trainDataProcessed.append(trainSet[list(instanceFeature)+["runtime_"+str(algIndex)]])

	# save training data in file to train each model.
	for i in range(len(trainDataProcessed)):
		##print "each algrithem training Set shape:",trainDataProcessed[i].shape
		trainDataProcessed[i].to_csv("regression_training_data/algorithm"+str(i)+"_training_data.csv",index=False)


	trainDataProcessed=[]
	for algIndex in range(len(algorithmNames)):
		trainDataProcessed.append(trainSet2[list(instanceFeature)+["runtime_"+str(algIndex)]])

	# save training data in file to train each model.
	for i in range(len(trainDataProcessed)):
		##print "each algrithem training Set shape:",trainDataProcessed[i].shape
		trainDataProcessed[i].to_csv("regression_training_data/algorithm"+str(i)+"_training_data2.csv",index=False)

	# save index to algorithm mapping
	with open("regression_training_data/index2algorithm.csv","w") as f:
		for element in index2Alg:
			f.write(element[0]+","+element[1]+"\n")


	####### classification data:
	#testing data
	if not os.path.isdir("classification_testing_data"):
		os.mkdir("classification_testing_data")
	testSet.to_csv("classification_testing_data/testing_data.csv",index=False)

	# we get training data for each algorithm
	if not os.path.isdir("classification_training_data"):
		os.mkdir("classification_training_data")

	trainSet,trainSet2=testbins(trainSet,bins)
	trainSet=pd.DataFrame(trainSet)
	trainSet2=pd.DataFrame(trainSet2)
	trainSet.to_csv("classification_training_data/training_data.csv",index=False)
	trainSet2.to_csv("classification_training_data/training_data2.csv",index=False)



#get instance running time for each algorithm
def getRunTmPerAlgo():

	algoRTFile="algorithm_runs.csv"
	pd1=pd.read_csv('./csv/'+algoRTFile)

	runTmPerAlgo=[]
	algorithmNames=list(set(pd1["algorithm"].values))
	algorithmNames=sorted(algorithmNames)

	index=0
	index2Alg=[]
	for algo in algorithmNames:
		singleAlg=pd1[pd1["algorithm"]==algo]
		#change index to instance_id
		singleAlg=singleAlg.set_index("instance_id")
		#only save instance_id and running time
		singleAlg=singleAlg[["runtime"]]
		#change "runtime" to "runtime_index" to distinguish different algrithms
		singleAlg.columns=["runtime_"+str(index)]
		#save the mapping of index to algorithm name
		index2Alg.append((str(index),algo))
		index+=1
		#print "Instance runtime shape for each algorithm:",algo,singleAlg.shape
		runTmPerAlgo.append(singleAlg)

	return algorithmNames,index2Alg,runTmPerAlgo


'''
#This is test case for   getRunTmPerAlgo()
algorithmNames,index2Alg,runTmPerAlgo=getRunTmPerAlgo()
#print algorithmNames
#print index2Alg
#print len(runTmPerAlgo),runTmPerAlgo[0].shape
'''

#get feature values
def getFeatureValue():
	featurFile="feature_values.csv"
	pd1=pd.read_csv('./csv/'+featurFile)
	pd1=pd1.set_index("instance_id")
	#print "Instence feature shape:",featurFile,pd1.shape
	return pd1

'''
#This is test case for   getFeatureValue()
pd1=getFeatureValue()
#print pd1.shape
'''


#Remove invalid values
#values that are "?"
#"repetition" columns
def dataRemoveInvalid(pd1):

	##print "original_shape",pd1.shape

	#delete column "repetition"

	#pd1=pd1.drop(["repetition"],axis=1)
	featureList=pd1.columns.values

	#drop "na" rows
	pd1=pd1.dropna(axis=0, how='any')

	#drop "?" rows
	for feature in featureList[1:]:
		if pd1[feature].dtypes=="object":
			# delete from the pd1 rows that contain "?"
			pd1=pd1[pd1[feature].astype("str")!="?"]
	##print "shape after preprocessing",pd1.shape
	return pd1

'''
#This is test case for   dataRemoveInvalid(pd1)
pd0=pd.DataFrame(data={"c1":[0,1,2],
			"c2":["1","2","?"]
			})

pd1=pd.DataFrame(data={"c3":["0","?","?"],
			})
pd2=pd.DataFrame(data={"c4":[0,1,2,4],
			"repetition":["1","2","3","4"]
			})
pd3=pd.concat([pd0,pd1,pd2],axis=1)
#print pd3
#print pd3.shape
#print dataRemoveInvalid(pd3)
'''


def regressionTrain(folder,filename):

	#get training data for each algorithm
	pd1=pd.read_csv(folder+filename)
	if Medium_diff == 1:
		pd1=pd1[pd1.iloc[:,-1]>30]
		pd1=pd1[pd1.iloc[:,-1]<TIME_MAX]
	X=pd1.iloc[:,:-1].values
	##print "Training set shape:",X.shape
	y=pd1.iloc[:,-1].values
	##print X.shape,y.shape
	##print "training data:",X.shape
	#for each algorithm,build 3 models
	#one is decision tree regression
	#another is random forest regession
	####Added: we add new model KNN regression



	scaler = StandardScaler()
	scaler.fit(X)

	if NORMALIZE==1:
		X=scaler.transform(X)


	bestDepthDT=0
	bestDepthRF=0
	bestKNeib=0

	bestDepth={}

	#Load parameter from pickle
	if os.path.isdir("parameter_pickle"):
		pickleFiles=[pickFile for pickFile in os.listdir('./parameter_pickle') if pickFile.endswith(".pickle")]
		if 'regression_bestDepth.pickle' in pickleFiles:
			with open('./parameter_pickle/regression_bestDepth.pickle', 'rb') as handle:
		 		bestDepth = pickle.load(handle)
	#
	#Here we show what the dictionary looks like
	#key: algorithm+index
	#value: (bestDepthDT,bestDepthRF,bestKNeib)
	'''
	bestDepth={'algorithm0': (9, 12,11), 'algorithm1': (4, 10,14), 'algorithm2': (5, 7,14),
		'algorithm3': (3, 5 ,10),'algorithm4': (5, 7, 14),'algorithm5': (2, 13, 14),
		'algorithm6': (5, 7, 4),'algorithm7': (4, 10,4),'algorithm8': (5, 10,4),
		'algorithm9': (3, 6,7),'algorithm10': (4, 5,7)
		 }
	'''

	#map filename to dictionary key value
	#For example, the filename is algorithm6_training_data, we map it to "algorithm6"
	fileKey=filename.split("_")[0]
	bestDepthDT,bestDepthRF,bestKNeib=bestDepth.get(fileKey,(0,0,0))




	#If the model is not trained before, we use cross validation to find best depth for it. # and k for kNN

	if bestKNeib==0 or bestDepthDT==0 or bestDepthRF==0:
		#print "Model is not pre-trained."
		#print "Use cross validation to find best depth and K..."
		#print "Pre-train model for ",fileKey,"..."
		max_depth = range(2, 30, 1)
		dt_scores = []
		for k in max_depth:
		    regr_k =tree.DecisionTreeRegressor(max_depth=k, random_state=1,min_samples_leaf=2, min_samples_split=2)
		    loss = -cross_val_score(regr_k, X, y, cv=10, scoring=score_f)
		    dt_scores.append(loss.mean())
		#print "DTscoring:",dt_scores
		plt.plot(max_depth, dt_scores,label="DT")
		plt.xlabel('Value of depth: Algorithm'+(filename.split("_")[0]))
		plt.ylabel('Cross-Validated MSE')
		#plt.show()
		bestscoreDT,bestDepthDT=sorted(list(zip(dt_scores,max_depth)))[0]
		##print "bestscoreDT:",bestscoreDT


		max_depth = range(2, 30, 1)
		dt_scores = []
		for k in max_depth:
		    regr_k = RandomForestRegressor(max_depth=k, random_state=1,min_samples_leaf=2, min_samples_split=2)
		    loss = -cross_val_score(regr_k, X, y, cv=10, scoring=score_f)
		    dt_scores.append(loss.mean())
		plt.plot(max_depth, dt_scores,label="RF")
		#print "RFscoring:",dt_scores
		plt.xlabel('Value of depth: Algorithm'+(filename.split("_")[0]))
		plt.ylabel('Cross-Validated MSE')
		#plt.show()
		bestscoreRF,bestDepthRF=sorted(list(zip(dt_scores,max_depth)))[0]
		##print "bestscoreRF:",bestscoreRF

		max_neigh = range(2, 30, 1)
		knn_scores = []
		for k in max_neigh:
		    kNeigh =KNeighborsRegressor(n_neighbors=k)
		    loss = -cross_val_score(kNeigh, X, y, cv=10, scoring=score_f)
		    knn_scores.append(loss.mean())
		#print "knnscoring:",knn_scores
		plt.plot(max_neigh, knn_scores,label="KNN")
		plt.xlabel('Value of depth: regression_'+(filename.split("_")[0]))
		plt.ylabel('Cross-Validated MSE')
		#plt.show()
		plt.legend()

		if not os.path.isdir("Best_Depth_Curve"):
			os.mkdir("Best_Depth_Curve")
		plt.savefig("Best_Depth_Curve/regression_"+filename.split("_")[0])
		plt.clf()
		bestscoreRF,bestKNeib=sorted(list(zip(knn_scores,max_neigh)))[0]
		##print "bestscoreRF:",bestscoreRF


		bestDepth[filename.split("_")[0]]=(bestDepthDT,bestDepthRF,bestKNeib)
	if not os.path.isdir("parameter_pickle"):
		os.mkdir("parameter_pickle")
	with open('parameter_pickle/regression_bestDepth.pickle', 'wb') as handle:
	    pickle.dump(bestDepth, handle)

	##print bestDepth


	#We build three models using best depth, trained with all data
	#print "Regression parameters for DT, RF, KNN-",filename.split("_")[0],"-",bestDepthDT,bestDepthRF,bestKNeib
	dtModel =tree.DecisionTreeRegressor(max_depth=bestDepthDT, random_state=1,min_samples_leaf=2, min_samples_split=2)
	dtModel = dtModel.fit(X, y)
	y_=dtModel.predict(X)
	#print "DT,average_relative_error:",-relative_score(y, y_),-max_relative_score(y, y_)

	if not os.path.isdir("training_result"):
		os.mkdir("training_result")
	pd_training_error_save=pd.read_csv(folder+filename)
	if Medium_diff == 1:
		pd_training_error_save=pd_training_error_save[pd_training_error_save.iloc[:,-1]>30]
		pd_training_error_save=pd_training_error_save[pd_training_error_save.iloc[:,-1]<TIME_MAX]
	pd_training_error_save["DT_training_pred"]=y_
	pd_training_error_save.to_csv("training_result/DT_training_pred"+filename.split("_")[0]+".csv")

	index2Algo_pd=pd.read_csv("regression_training_data/index2algorithm.csv",header=None)

	pd_temp=pd.read_csv("regression_training_data/algorithm"+str(0)+"_training_data.csv")
	for alg_index in range(1,len(index2Algo_pd)):
		pd_temp1=pd.read_csv("regression_training_data/algorithm"+str(alg_index)+"_training_data.csv")
		pd_temp["runtime_"+str(alg_index)]=pd_temp1["runtime_"+str(alg_index)]
	pd_temp.to_csv("regression_training_data/training_data.csv",index=False)


	pd_temp=pd.read_csv("regression_training_data/algorithm"+str(0)+"_training_data2.csv")
	for alg_index in range(1,len(index2Algo_pd)):
		pd_temp1=pd.read_csv("regression_training_data/algorithm"+str(alg_index)+"_training_data2.csv")
		pd_temp["runtime_"+str(alg_index)]=pd_temp1["runtime_"+str(alg_index)]
	pd_temp.to_csv("regression_training_data/training_data2.csv",index=False)

	write2file(filename.split("_")[0])

	avgruntime_=sum(y)/float(len(y))
	numof200_=sum([0 if v>TIME_MAX-1 else 1 for v in y])
	solved_=numof200_/float(len(y))

	write2file(str(solved_)+"/"+str(avgruntime_))





	write2file(str(-relative_score(y, y_))+"  /  "+str(-max_relative_score(y, y_))+"  /  "+str(mean_squared_error(y, y_)**0.5))
	rfModel = RandomForestRegressor(max_depth=bestDepthDT, random_state=1,min_samples_leaf=2, min_samples_split=2)
	rfModel = rfModel.fit(X, y)
	y_=rfModel.predict(X)
	#print "RF,average_relative_error:",-relative_score(y, y_),-max_relative_score(y, y_)

	pd_training_error_save=pd.read_csv(folder+filename)
	pd_training_error_save["RF_training_pred"]=y_
	pd_training_error_save.to_csv("training_result/RF_training_pred"+filename.split("_")[0]+".csv")


	write2file(str(-relative_score(y, y_))+"  /  "+str(-max_relative_score(y, y_))+"  /  "+str(mean_squared_error(y, y_)**0.5))
	kNeigh =KNeighborsRegressor(n_neighbors=bestKNeib)
	kNeigh =kNeigh.fit(X,y)
	y_=kNeigh.predict(X)
	#print "kNN ,average_relative_error:",-relative_score(y, y_),-max_relative_score(y, y_)

	pd_training_error_save=pd.read_csv(folder+filename)
	pd_training_error_save["kNN_training_pred"]=y_
	pd_training_error_save.to_csv("training_result/kNN_training_pred"+filename.split("_")[0]+".csv")


	write2file(str(-relative_score(y, y_))+"  /  "+str(-max_relative_score(y, y_))+"  /  "+str(mean_squared_error(y, y_)**0.5),1)
	return dtModel,rfModel,kNeigh,scaler


'''
#Test if we can mannually load parameter
bestDepth={'algorithm0': (9, 12,11), 'algorithm1': (4, 10,14), 'algorithm2': (5, 7,14),
		'algorithm3': (3, 5 ,10),'algorithm4': (5, 7, 14),'algorithm5': (2, 13, 14),
		'algorithm6': (5, 7, 4),'algorithm7': (4, 10,4),'algorithm8': (5, 10,4),
		'algorithm9': (3, 6,7),'algorithm10': (4, 5,7)
		 }

if not os.path.isdir("parameter_pickle"):
		os.mkdir("parameter_pickle")
with open('parameter_pickle/bestDepth.pickle', 'wb') as handle:
	  	pickle.dump(bestDepth, handle)

'''
def regressionValidate(solvernames):


	#Since we have already got our test file, we use it to test our model.
	test_pd=pd.read_csv("regression_training_data/training_data2.csv")


	index2Algo_pd=pd.read_csv("regression_training_data/index2algorithm.csv",header=None)
	#test_x, test_y
	test_x=test_pd.iloc[:,:-index2Algo_pd.shape[0]].values
	#print "validation data:",test_x.shape
	testcopy=test_x.copy()
	test_x=[i.reshape(1,-1) for i in test_x]

	test_y_list=test_pd.iloc[:,-index2Algo_pd.shape[0]:].values
	test_y=[smallestValueIndex(runtimeList) for runtimeList in test_y_list]

	#build three models for each algorithm
	algoFileNames=["algorithm"+str(i)+"_training_data.csv" for i in range(len(index2Algo_pd))]



	write2file("accuracy,")

	modelNames=["DT","RF","KNN"]
	predictionof3Model=[]
	for i in range(len(modelNames)):
		predictionEachModel=[]

		hams_pred={}
		for j in range(len(algoFileNames)):
			hams_pred[j]=[]


		for each_inst in test_x:
			#resultList=[solvername[i].predict(each_inst) for solvername in solvernames[:-1]]
			resultList=[solvername[i].predict(each_inst) for solvername in solvernames]
			##print "************",len(resultList)
			for m in range(len(algoFileNames)):
				hams_pred[m].append(resultList[m][0])

			pred=smallestValueFirst3(resultList)
			predictionEachModel.append(pred)
			'''###
			pred=smallestValueIndex(resultList)
			prediction.append(pred)
			'''
		#calculate indepedent accuracy.
		#print modelNames[i],'Model Accuracy is %5.3f%%.' % (getAccuracyFirst3(predictionEachModel,test_y)*100)

		write2file(str(getAccuracyFirst3(predictionEachModel,test_y)))

		savevalidatonRegPred(hams_pred,predictionEachModel,modelNames[i])

		####print modelNames[i],'Model Accuracy is %5.3f%%.' % (getAccuracy(prediction,test_y)*100)
		predictionof3Model.append(predictionEachModel) #3*n*top3

	prediction3ModelTop9=[]#n*(top 9)
	for i in range(len(predictionof3Model[0])):
		predic_instance=list(predictionof3Model[0][i])+list(predictionof3Model[1][i])+list(predictionof3Model[2][i])
		prediction3ModelTop9.append(predic_instance)

	###predictionofThree=np.array(predictionofThree).T
	###prediction=[majorityVotingPred(predict) for predict in predictionofThree]

	prediction=[majorityVotingPredTop3(predict) for predict in prediction3ModelTop9]

	#calculate majVoting accuracy.
	#print 'Voting Accuracy is %5.3f%%.' % (getAccuracyFirst3(prediction,test_y)*100)
	####print 'Voting Accuracy is %5.3f%%.' % (getAccuracy(prediction,test_y)*100)
	#print "\n"
	#write2file(str(getAccuracyFirst3(prediction,test_y)),1)
	write2file("",1)
	getmodelperformance()
	getsoa()



def getmodelperformance():

	perf_dic={}
	for f in ["DT","RF","KNN"]:
		perf_dic[f]=[]
		df2=pd.read_csv("regression_valid_result/"+f+"_pre.csv")

		sol=f+"_pre_runtime_1"
		df3 = df2[sol].values
		avgruntime=sum(df3)/float(len(df3))
		numof600=sum([0 if v>TIME_MAX-1 else 1 for v in df3])
		solved=numof600/float(len(df3))
		##print sol,", 1st solver, solved, avgruntime,",solved,",",avgruntime

		perf_dic[f].append(str(solved)+"/"+str(avgruntime))

		sol=f+"_pre_runtime_2"
		df3 = df2[sol].values
		avgruntime=sum(df3)/float(len(df3))
		numof600=sum([0 if v>TIME_MAX-1 else 1 for v in df3])
		solved=numof600/float(len(df3))
		##print sol,", 2nd solver, solved, avgruntime,",solved,",",avgruntime
		perf_dic[f].append(str(solved)+"/"+str(avgruntime))


		sol=f+"_pre_runtime_3"
		df3 = df2[sol].values
		avgruntime=sum(df3)/float(len(df3))
		numof600=sum([0 if v>TIME_MAX-1 else 1 for v in df3])
		solved=numof600/float(len(df3))
		##print sol,", 3rd solver, solved, avgruntime,",solved,",",avgruntime
		perf_dic[f].append(str(solved)+"/"+str(avgruntime))

		fulltime=TIME_MAX
		fiexedRT=50 # set 100 seconds to run 3 algorithms
		df3 = df2.loc[:,[f+"_pre_runtime_1",f+"_pre_runtime_2",f+"_pre_runtime_3"]].values
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

		#timeForPred=[1000 if v>1000 else v for v in timeForPred]
		avgruntime=sum(timeForPred)/float(len(timeForPred))
		numof600=sum([0 if v>TIME_MAX-1 else 1 for v in timeForPred])
		solved=numof600/float(len(timeForPred))
		perf_dic[f].append(str(solved)+"/"+str(avgruntime))
	write2file("1st,")
	for f in ["DT","RF","KNN"]:
		write2file(perf_dic[f][0])
	write2file("",1)

	write2file("2nd,")
	for f in ["DT","RF","KNN"]:
		write2file(perf_dic[f][1])
	write2file("",1)

	write2file("3rd,")
	for f in ["DT","RF","KNN"]:
		write2file(perf_dic[f][2])
	write2file("",1)
	write2file("portfolio,")
	for f in ["DT","RF","KNN"]:
		write2file(perf_dic[f][3])
	write2file("",1)



def getsoa():
	write2file("soa")
	f=pd.read_csv("regression_training_data/training_data2.csv")
	alg=["runtime_"+str(i) for i in range(6)]
	alg=f.loc[:,alg].values

	def getmin(alg):
		ret=[]
		for i in range(len(alg)):
			ret.append(min(alg[i]))
		return ret

	l=getmin(alg)
	write2file(str(sum([1 if i<TIME_MAX-1 else 0 for i in l])/float(len(l)))+"/"+str(sum(l)/float(len(l))),1)




def regressionPredict():

	if os.path.isdir("regression_training_data") and os.path.isdir("regression_testing_data"):
		trainingCsvFiles=[filename for filename in os.listdir('./regression_training_data') if filename.endswith(".csv")]
		testingCsvFiles=[filename for filename in os.listdir('./regression_testing_data') if filename.endswith(".csv")]

		if not "algorithm0_training_data.csv" in trainingCsvFiles or not "testing_data.csv" in testingCsvFiles:
			#print "No training and testing data "
			return 0


	#Since we have already got our test file, we use it to test our model.
	test_pd=pd.read_csv("regression_testing_data/testing_data.csv")
	index2Algo_pd=pd.read_csv("regression_training_data/index2algorithm.csv",header=None)

	#test_x, test_y
	test_x=test_pd.iloc[:,:-index2Algo_pd.shape[0]].values
	#print "testing data:",test_x.shape
	testcopy=test_x.copy()
	test_x=[i.reshape(1,-1) for i in test_x]

	test_y_list=test_pd.iloc[:,-index2Algo_pd.shape[0]:].values
	test_y=[smallestValueIndex(runtimeList) for runtimeList in test_y_list]

	#build three models for each algorithm
	algoFileNames=["algorithm"+str(i)+"_training_data.csv" for i in range(len(index2Algo_pd))]

	write2file("Regression(Average_relative_error),,Average_relative_error/Max_absolute_error/Root Mean Squared Error",1)
	write2file("Validation accuracy",1)
	write2file("Alg,solved/time,DT,RF,KNN",1)


	solvernames=[regressionTrain("regression_training_data/",alg) for alg in algoFileNames]
	if NORMALIZE==1:
	    test_x=solvernames[0][-1].transform(testcopy)
	    test_x=[i.reshape(1,-1) for i in test_x]
	#predict indepedently and using majority voting
	#print "Regression result:"

	regressionValidate(solvernames)

	write2file("Testing accuracy",1)
	write2file("Alg,solved/time,DT,RF,KNN",1)


	allacurracys=[]
	modelNames=["DT","RF","KNN"]
	predictionof3Model=[]
	for i in range(len(modelNames)):
		predictionEachModel=[]

		hams_pred={}
		for j in range(len(algoFileNames)):
			hams_pred[j]=[]


		for each_inst in test_x:
			#resultList=[solvername[i].predict(each_inst) for solvername in solvernames[:-1]]
			resultList=[solvername[i].predict(each_inst) for solvername in solvernames]
			##print "************",len(resultList)
			for m in range(len(algoFileNames)):
				hams_pred[m].append(resultList[m][0])

			pred=smallestValueFirst3(resultList)
			predictionEachModel.append(pred)
			'''###
			pred=smallestValueIndex(resultList)
			prediction.append(pred)
			'''
		#calculate indepedent accuracy.
		accuracy_i=getAccuracyFirst3(predictionEachModel,test_y)
		allacurracys.append(accuracy_i)
		#print modelNames[i],'Model Accuracy is %5.3f%%.' % (accuracy_i*100)

		#write2file(str(getAccuracyFirst3(predictionEachModel,test_y)*100))

		saveRegPred(hams_pred,predictionEachModel,modelNames[i])
		####print modelNames[i],'Model Accuracy is %5.3f%%.' % (getAccuracy(prediction,test_y)*100)
		predictionof3Model.append(predictionEachModel) #3*n*top3

	prediction3ModelTop9=[]#n*(top 9)
	for i in range(len(predictionof3Model[0])):
		predic_instance=list(predictionof3Model[0][i])+list(predictionof3Model[1][i])+list(predictionof3Model[2][i])
		prediction3ModelTop9.append(predic_instance)

	###predictionofThree=np.array(predictionofThree).T
	###prediction=[majorityVotingPred(predict) for predict in predictionofThree]

	prediction=[majorityVotingPredTop3(predict) for predict in prediction3ModelTop9]

	#calculate majVoting accuracy.
	#print 'Voting Accuracy is %5.3f%%.' % (getAccuracyFirst3(prediction,test_y)*100)
	####print 'Voting Accuracy is %5.3f%%.' % (getAccuracy(prediction,test_y)*100)
	#print "\n"
	#write2file(str(getAccuracyFirst3(prediction,test_y)*100),1)

	allresult={}
	for model_name in ["DT","RF","KNN"]:
		df=pd.read_csv("regression_result/"+model_name+"_pre.csv")
		for alg in range(6):
			y_true=df["runtime_"+str(alg)]
			avgruntime_=sum(y_true)/float(len(y_true))
			numof200_=sum([0 if v>TIME_MAX-1 else 1 for v in y_true])
			solved_=numof200_/float(len(y_true))
			allresult[str(alg)]=str(solved_)+"/"+str(avgruntime_)
			y_pred=df[model_name+"_model_value"+str(alg)]
			allresult[model_name+str(alg)]="/".join([str(-relative_score(y_true, y_pred)),str(-max_relative_score(y_true, y_pred)),str(mean_squared_error(y_true, y_pred)**0.5)])

	for i in range(6):
		write2file("algorithm"+str(i))
		write2file(allresult[str(i)])
		for model_name in ["DT","RF","KNN"]:
			write2file(allresult[model_name+str(i)])

		write2file("",1)

	write2file("accuracy,")
	write2file(str(allacurracys[0]))
	write2file(str(allacurracys[1]))
	write2file(str(allacurracys[2]),1)
	getmodelperformance2()
	getsoa2()



def getsoa2():
	write2file("soa")
	f=pd.read_csv("regression_testing_data/testing_data.csv")
	alg=["runtime_"+str(i) for i in range(6)]
	alg=f.loc[:,alg].values

	def getmin(alg):
		ret=[]
		for i in range(len(alg)):
			ret.append(min(alg[i]))
		return ret

	l=getmin(alg)
	write2file(str(sum([1 if i<TIME_MAX-1 else 0 for i in l])/float(len(l)))+"/"+str(sum(l)/float(len(l))),1)

def getmodelperformance2():

	perf_dic={}
	for f in ["DT","RF","KNN"]:
		perf_dic[f]=[]
		df2=pd.read_csv("regression_result/"+f+"_pre.csv")

		sol=f+"_pre_runtime_1"
		df3 = df2[sol].values
		avgruntime=sum(df3)/float(len(df3))
		numof600=sum([0 if v>TIME_MAX-1 else 1 for v in df3])
		solved=numof600/float(len(df3))
		##print sol,", 1st solver, solved, avgruntime,",solved,",",avgruntime

		perf_dic[f].append(str(solved)+"/"+str(avgruntime))

		sol=f+"_pre_runtime_2"
		df3 = df2[sol].values
		avgruntime=sum(df3)/float(len(df3))
		numof600=sum([0 if v>TIME_MAX-1 else 1 for v in df3])
		solved=numof600/float(len(df3))
		##print sol,", 2nd solver, solved, avgruntime,",solved,",",avgruntime
		perf_dic[f].append(str(solved)+"/"+str(avgruntime))


		sol=f+"_pre_runtime_3"
		df3 = df2[sol].values
		avgruntime=sum(df3)/float(len(df3))
		numof600=sum([0 if v>TIME_MAX-1 else 1 for v in df3])
		solved=numof600/float(len(df3))
		##print sol,", 3rd solver, solved, avgruntime,",solved,",",avgruntime
		perf_dic[f].append(str(solved)+"/"+str(avgruntime))

		fulltime=TIME_MAX
		fiexedRT=50 # set 100 seconds to run 3 algorithms
		df3 = df2.loc[:,[f+"_pre_runtime_1",f+"_pre_runtime_2",f+"_pre_runtime_3"]].values
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

		#timeForPred=[1000 if v>1000 else v for v in timeForPred]
		avgruntime=sum(timeForPred)/float(len(timeForPred))
		numof600=sum([0 if v>TIME_MAX-1 else 1 for v in timeForPred])
		solved=numof600/float(len(timeForPred))
		perf_dic[f].append(str(solved)+"/"+str(avgruntime))
	write2file("1st,")
	for f in ["DT","RF","KNN"]:
		write2file(perf_dic[f][0])
	write2file("",1)

	write2file("2nd,")
	for f in ["DT","RF","KNN"]:
		write2file(perf_dic[f][1])
	write2file("",1)

	write2file("3rd,")
	for f in ["DT","RF","KNN"]:
		write2file(perf_dic[f][2])
	write2file("",1)
	write2file("portfolio,")
	for f in ["DT","RF","KNN"]:
		write2file(perf_dic[f][3])
	write2file("",1)




def getAccuracy(y1,y2):
	len1=len(y1)
	len2=len(y2)
	if len1==len2:
		same=0.0
		for index in range(len1):
			if y1[index]==y2[index]:
				same+=1
		return same/len1
	else:
		#print "The two lists are not of same length"
		return 0

def majorityVotingPredTop3(predList):
	#Count appearance and sort them
	counterPred=Counter(predList)
	sortPred=sorted([(counterPred[key],key) for key in counterPred])
	return [sortPred[-1][1],sortPred[-2][1],sortPred[-3][1]]


def majorityVotingPred(predList):
	#Count appearance and sort them
	counterPred=Counter(predList)
	sortPred=sorted([(counterPred[key],key) for key in counterPred])

	# if the max appearance only appears once, use DT.
	if sortPred[-1][0]==1:
		return predList[2]

	else:# use Maj voting
		return sortPred[-1][1]


'''
#test case for  majorityVotingPred()
predList1=[1,3,2,1]
predList2=[1,3,2,4]
#print majorityVotingPred(predList1)
#print majorityVotingPred(predList2)
'''

def getAccuracyFirst3(y1,y2):
	len1=len(y1)
	len2=len(y2)
	if len1==len2:
		same=0.0
		for index in range(len1):
			if y2[index] in y1[index]:
				same+=1
		return same/len1
	else:
		#print "The two lists are not of same length"
		return 0


def smallestValueFirst3(resultList):
	result=[(time,i) for i,time in enumerate(resultList)]
	algOrder=[index for time,index in sorted(result)]
	return algOrder[:3]


def smallestValueIndex(resultList):
	result=[(time,i) for i,time in enumerate(resultList)]
	algOrder=[index for time,index in sorted(result)]
	return algOrder[0]
'''
#test case for  smallestValueIndex()
predList1=[12,32,0.52,4.5]
#print smallestValueIndex(predList1)
'''

def savevalidatonRegPred(ham_pre_dic,predictionTop3list,name):
	p1=[tuples[0] for tuples in predictionTop3list]
	p2=[tuples[1] for tuples in predictionTop3list]
	p3=[tuples[2] for tuples in predictionTop3list]

	test_pd1=pd.read_csv("regression_training_data/training_data2.csv")
	for i in range(len(ham_pre_dic.keys())):
		test_pd1[name+"_model_value"+str(i)]=ham_pre_dic[i]
	test_pd1[name+"_pre_1"]=p1
	test_pd1[name+"_pre_runtime_1"]=solver2timeReg_training(p1)
	test_pd1[name+"_pre_2"]=p2
	test_pd1[name+"_pre_runtime_2"]=solver2timeReg_training(p2)
	test_pd1[name+"_pre_3"]=p3
	test_pd1[name+"_pre_runtime_3"]=solver2timeReg_training(p3)

	if not os.path.isdir("regression_valid_result"):
		os.mkdir("regression_valid_result")
	test_pd1.to_csv("regression_valid_result/"+name+"_pre.csv",index=False)



def saveRegPred(ham_pre_dic,predictionTop3list,name) :
	p1=[tuples[0] for tuples in predictionTop3list]
	p2=[tuples[1] for tuples in predictionTop3list]
	p3=[tuples[2] for tuples in predictionTop3list]

	test_pd1=pd.read_csv("regression_testing_data/testing_data.csv")
	for i in range(len(ham_pre_dic.keys())):
		test_pd1[name+"_model_value"+str(i)]=ham_pre_dic[i]
	test_pd1[name+"_pre_1"]=p1
	test_pd1[name+"_pre_runtime_1"]=solver2timeReg(p1)
	test_pd1[name+"_pre_2"]=p2
	test_pd1[name+"_pre_runtime_2"]=solver2timeReg(p2)
	test_pd1[name+"_pre_3"]=p3
	test_pd1[name+"_pre_runtime_3"]=solver2timeReg(p3)

	if not os.path.isdir("regression_result"):
		os.mkdir("regression_result")
	test_pd1.to_csv("regression_result/"+name+"_pre.csv",index=False)


def solver2timeReg_training(solverlist):

        #index2Algo_pd=pd.read_csv("regression_training_data/index2algorithm.csv",header=None)

	test_pd1=pd.read_csv("regression_training_data/training_data2.csv")
	#solverlist=[1]*test_pd1.shape[0]
	#solverlist[0]=0
	timelist=[test_pd1.iloc[i,:]["runtime_"+str(solver)] for i,solver in enumerate(solverlist)]
	return timelist


def solver2timeReg(solverlist):

        #index2Algo_pd=pd.read_csv("regression_training_data/index2algorithm.csv",header=None)

	test_pd1=pd.read_csv("regression_testing_data/testing_data.csv")
	#solverlist=[1]*test_pd1.shape[0]
	#solverlist[0]=0
	timelist=[test_pd1.iloc[i,:]["runtime_"+str(solver)] for i,solver in enumerate(solverlist)]
	return timelist



def classifierTrain(folder,filename):


	#get training data
	trainPd=pd.read_csv(folder+filename)
	index2Algo_pd=pd.read_csv("regression_training_data/index2algorithm.csv",header=None)
	algorithmSize=index2Algo_pd.shape[0]

	X=np.array(list(trainPd.iloc[:,:-algorithmSize].values))
	X=X.reshape(X.shape[0],-1)
	#print "training data:",X.shape

	scaler1 = StandardScaler()
	scaler1.fit(X)
	if NORMALIZE==1:
		X=scaler1.transform(X)


	y=trainPd.iloc[:,-algorithmSize:].values
	y=[smallestValueIndex(runtimeList) for runtimeList in y]
	y=np.array(y).reshape(len(y),)
	#build 3 models
	#one is decision tree regression
	#another is random forest regession
	####Added: we add new model KNN regression

	bestDepthDT=0
	bestDepthRF=0
	bestKNeib=0

	bestDepth={}

	#Load parameter from pickle
	if os.path.isdir("parameter_pickle"):
		pickleFiles=[pickFile for pickFile in os.listdir('./parameter_pickle') if pickFile.endswith(".pickle")]
		if 'classifier_bestDepth.pickle' in pickleFiles:
			with open('./parameter_pickle/classifier_bestDepth.pickle', 'rb') as handle:
		 		bestDepth = pickle.load(handle)
	#
	#Here we show what the dictionary looks like
	#key: algorithm+index
	#value: (bestDepthDT,bestDepthRF,bestKNeib)

	#bestDepth={'training_data': (9, 12,11)}


	#map filename to dictionary key value
	#For example, the filename is algorithm6_training_data, we map it to "algorithm6"
	fileKey=filename.split(".")[0]
	bestDepthDT,bestDepthRF,bestKNeib=bestDepth.get(fileKey,(0,0,0))


	#If the model is not trained before, we use cross validation to find best depth for it. # and k for kNN

	if bestKNeib==0 or bestDepthDT==0 or bestDepthRF==0:
		#print "Model is not pre-trained."
		#print "Use cross validation to find best depth and K..."
		#print "Pre-train model for ",fileKey,"..."
		max_depth = range(2, 15, 1)
		dt_scores = []
		for k in max_depth:
		    regr_k =tree.DecisionTreeClassifier(max_depth=k, random_state=1,min_samples_leaf=2, min_samples_split=2)
		    loss = -cross_val_score(regr_k, X, y, cv=10, scoring='accuracy')
		    dt_scores.append(loss.mean())
		plt.plot(max_depth, dt_scores,label="DT")
		plt.xlabel('Value of depth: Algorithm'+(filename.split(".")[0]))
		plt.ylabel('Cross-Validated MSE')
		#plt.show()
		bestDepthDT=sorted(list(zip(dt_scores,max_depth)))[0][1]


		max_depth = range(2, 15, 1)
		dt_scores = []
		for k in max_depth:
		    regr_k = RandomForestClassifier(max_depth=k, random_state=1,min_samples_leaf=2, min_samples_split=2)
		    loss = -cross_val_score(regr_k, X, y, cv=10, scoring='accuracy')
		    dt_scores.append(loss.mean())
		plt.plot(max_depth, dt_scores,label="RF")
		plt.xlabel('Value of depth: Algorithm'+(filename.split(".")[0]))
		plt.ylabel('Cross-Validated MSE')
		#plt.show()
		bestDepthRF=sorted(list(zip(dt_scores,max_depth)))[0][1]


		max_neigh = range(2, 15, 1)
		knn_scores = []
		for k in max_neigh:
		    kNeigh =KNeighborsClassifier(n_neighbors=k)
		    loss = -cross_val_score(kNeigh, X, y, cv=10, scoring='accuracy')
		    knn_scores.append(loss.mean())
		plt.plot(max_neigh, knn_scores,label="KNN")
		plt.xlabel('Value of depth: classifier')
		plt.ylabel('Cross-Validated MSE')
		#plt.show()
		plt.legend()

		if not os.path.isdir("Best_Depth_Curve"):
			os.mkdir("Best_Depth_Curve")
		plt.savefig("Best_Depth_Curve/classifier")
		plt.clf()
		bestKNeib=sorted(list(zip(knn_scores,max_neigh)))[0][1]

		bestDepth[filename.split(".")[0]]=(bestDepthDT,bestDepthRF,bestKNeib)
	if not os.path.isdir("parameter_pickle"):
		os.mkdir("parameter_pickle")
	with open('parameter_pickle/classifier_bestDepth.pickle', 'wb') as handle:
	    pickle.dump(bestDepth, handle)

	##print bestDepth

	#We build three models using best depth, trained with all data

	##print len(Xcopy[0]),len(ycopy[0])

	#print "Classification parameters for DT, RF, KNN-",filename.split(".")[0],"-",bestDepthDT,bestDepthRF,bestKNeib
	dtModel =tree.DecisionTreeClassifier(max_depth=bestDepthDT, random_state=1,min_samples_leaf=2, min_samples_split=2)
	dtModel = dtModel.fit(X, y)


	#write2file("\nClassification",1)
	#write2file("Accuracy",1)
	#write2file("DT,RF,KNN",1)

	y_=dtModel.predict(X)
	#print "DT,accuracy_score:",accuracy_score(y, y_)
	#write2file(str(100*accuracy_score(y, y_)))

	rfModel = RandomForestClassifier(max_depth=bestDepthRF, random_state=1,min_samples_leaf=2, min_samples_split=2)
	rfModel = rfModel.fit(X, y)

	y_=rfModel.predict(X)
	#print "RF,accuracy_score:",accuracy_score(y, y_)
	#write2file(str(100*accuracy_score(y, y_)))

	kNeigh =KNeighborsClassifier(n_neighbors=bestKNeib)
	kNeigh =kNeigh.fit(X,y)

	y_=kNeigh.predict(X)
	#print "KNN,accuracy_score:",accuracy_score(y, y_)
	#write2file(str(100*accuracy_score(y, y_)),1)
	return dtModel,rfModel,kNeigh,scaler1

def classifierPredict():

	if os.path.isdir("classification_training_data") and os.path.isdir("classification_testing_data"):
		trainingCsvFiles=[filename for filename in os.listdir('./classification_training_data') if filename.endswith(".csv")]
		testingCsvFiles=[filename for filename in os.listdir('./classification_testing_data') if filename.endswith(".csv")]

		if not "training_data.csv" in trainingCsvFiles or not "testing_data.csv" in testingCsvFiles:
			#print "No training and testing data "
			return 0

	#Since we have already got our test file, we use it to test our model.
	test_pd=pd.read_csv("classification_testing_data/testing_data.csv")
	index2Algo_pd=pd.read_csv("regression_training_data/index2algorithm.csv",header=None)
	algorithmSize=index2Algo_pd.shape[0]

	test_x=np.array(list(test_pd.iloc[:,:-algorithmSize].values))
	test_x=test_x.reshape(test_x.shape[0],-1)
	test_y_list=test_pd.iloc[:,-algorithmSize:].values
	test_y=[smallestValueIndex(runtimeList) for runtimeList in test_y_list]
	test_y=np.array(test_y).reshape(len(test_y),)

	#build three models

	model=classifierTrain("classification_training_data/","training_data.csv")
	if NORMALIZE==1:
		test_x=model[-1].transform(test_x)

	#predict indepedently and using majority voting

	#print "Classifiers result:"
	#write2file("Classifiers result:",1)
	#write2file("DT,RF,KNN,VOTING",1)
	modelNames=["DT","RF","KNN"]
	predictionofThree=[]
	for i in range(len(modelNames)):


		prediction=model[i].predict(test_x)
		savePred(prediction,solver2time(prediction),modelNames[i])
		#calculate indepedent accuracy.
		#print modelNames[i],'Model Accuracy is %5.3f%%.' % (getAccuracy(prediction,test_y)*100)
		#write2file(str(getAccuracy(prediction,test_y)*100))
		predictionofThree.append(prediction)

	predictionofThree=np.array(predictionofThree).T
	prediction=[majorityVotingPred(predict) for predict in predictionofThree]
	savePred(prediction,solver2time(prediction),"Voting")
	#calculate majVoting accuracy.
	#print 'Voting Accuracy is %5.3f%%.' % (getAccuracy(prediction,test_y)*100)
	#write2file(str(getAccuracy(prediction,test_y)*100),1)

def solver2time(solverlist):
	##print "\n\nsolverlist",solverlist
	test_pd1=pd.read_csv("classification_testing_data/testing_data.csv")
	#solverlist=[1]*test_pd1.shape[0]
	#solverlist[0]=0
	timelist=[test_pd1.iloc[i,:]["runtime_"+str(solver)] for i,solver in enumerate(solverlist)]
	return timelist


def savePred(prediction,predictionTime,modelname):
	test_pd1=pd.read_csv("classification_testing_data/testing_data.csv")
	test_pd1[modelname+"_pre"]=prediction
	test_pd1[modelname+"_pre_runtime"]=predictionTime
	if not os.path.isdir("classification_result"):
		os.mkdir("classification_result")
	test_pd1.to_csv("classification_result/"+modelname+"_pre.csv",index=False)


##print solver2time([1])

def main():
	getTrainingData()
	regressionPredict()
	classifierPredict()
main()
