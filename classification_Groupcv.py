import numpy as np
import scipy.io 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
import logging

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC




#Participant dictionary keys: "P1", "P2", "P3", "P4", 'P5', 'P6', 'P7', 'P9', 'P10', 'P11', 'P12'
def load_data(subject_num):

	from scipy.signal import butter, lfilter
    
	#For 5-way classification
	filename = "data/P"+str(subject_num)+'dataxepoch_ar.mat'
	reject = [1,3,8,11,13]

	#skip noisy data
	if subject_num in reject :
		print("Noisy data.. choose a different participant")
	

	
	data = scipy.io.loadmat(filename)
	keys = [x for x in data.keys() if "P" in x]

	#create dictionary for each key for each participant
	eeg_data = []
	eeg_labels = []
	eeg_baseline = []
	
	#Channel selection 
	#F3, C3, P3, O1, Pz, FPz, Fz, F4, Cz, C4, P4, O2
	# channel_list = [4,12,20,26,30,32,37,39,47,49,57,63]
	#FC1,FCz,FC2,C2,Cz,C1,CP1,CPz,CP2,O1,O2,Oz
	channel_list = [14,12,47,49,51,37,30]
	
	for i in range(len(keys)):
		#start 205 for baseline
		# data_sub = data[keys[i]][0:100,channel_list,205:2000]
		data_sub = data[keys[i]][:,:,205:]
		baseline_data = data[keys[i]][0:100,:,0:204]
		#uncomment for erp-based features
		# b,a = butter(4, [0.1, 40], fs=2048, btype='band')
		# data_sub = lfilter(b, a, data_sub)

		
		eeg_data.extend(data_sub)
		eeg_baseline.extend(baseline_data)

		
	#Create group numberings
	initial_value = 1  
	repeat_count = 5  
	increment_value = 1  
	total_elements = np.shape(eeg_data)[0] 
	remainder = total_elements%5
	groups=[]


	for i in range (0,total_elements,repeat_count):
		if i == total_elements-(remainder):
				groups.extend([initial_value]*remainder)
		else:
				groups.extend([initial_value]*5)
		initial_value += increment_value
	trial_within_block = []
	for i in range(0,len(data),5):
		trial_within_block.extend([1,2,3,4,5])
	


	groups = np.asarray(groups)
	eeg_labels = np.concatenate((1*np.ones(np.shape(data[keys[0]])[0]),2*np.ones(np.shape(data[keys[1]])[0]),
							   3*np.ones(np.shape(data[keys[2]])[0]),4*np.ones(np.shape(data[keys[3]])[0]),
							   5*np.ones(np.shape(data[keys[4]])[0])))


	eeg_data = np.asarray(eeg_data)

	eeg_shuff, label_shuff,groups = shuffle_arrays(eeg_data, eeg_labels, groups, seed=42)


	return eeg_shuff, label_shuff, groups



def shuffle_arrays(A, B, C,seed=42):
    if seed is not None:
        np.random.seed(seed)
        
    # Generate a random permutation of indices for the first dimension
    perm = np.random.permutation(A.shape[0])
    
    # Apply the permutation to shuffle A and B
    A_shuffled = A[perm, ...]
    B_shuffled = B[perm, ...]
    C_shuffled = C[perm, ...]
    
    return A_shuffled, B_shuffled, C_shuffled





def train_classification_withholdout(data,labels,groups,clf):
	from sklearn.model_selection import cross_val_predict,GroupKFold
	from sklearn.decomposition import PCA
	from sklearn.pipeline import Pipeline
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.svm import SVC
	
	
	from mne.decoding import CSP
	import time
	
	start_time = time.time()

	n_feature_list = []
	n_feature_importance = []
	classification_scores = []  
	overall_conf_mat = np.zeros((5,5))
	score_matrix = np.zeros((4,5))



	logging.info("Entered classifier training (with holdout)..................................")
	
	
	data = np.asarray(data)
	data = data.reshape(data.shape[0],-1)

	
	
	logging.info("Initializing feature selector........................................................")
	#Feature Selection
	feature_sele = PCA(n_components=0.9)



	param = [
	{
			"n_estimators":[50,100,150,200,300],
			"max_depth":[3,4,5],
			"max_features":["log2","sqrt"],
	},
	{
		"C":[100,150,200,250,300,350,400,450],
		"gamma":[0.1, 1, 10, 100],
		"degree":[0,1,2,3],
		"kernel":[ 'poly','linear','rbf']
	},
	{
		"n_neighbors":[9,11,27,21],
		"weights":["distance"],
		"metric":["euclidean","cosine"]
	},
	{
		"solver":["svd","eigen"],
		"shrinkage":['auto']
		
	},
	{
 		'max_depth': [10, 20, 30, 40, 50, 60],
		'min_samples_leaf': [1, 2, 4],
		'min_samples_split': [2, 5, 6,7, 10],
		'n_estimators': [50,100,200]
	}]

	clf_param_map = {
		GradientBoostingClassifier: param[0],
		SVC: param[1],
		KNeighborsClassifier: param[2],
		LinearDiscriminantAnalysis: param[3],
		RandomForestClassifier: param[4],
	}

	logging.info("Initializing Classifier........................................................")
	
	logging.info("Initializing grid search parameters.............................................")
	#Grid Search CV
	grid_search = GridSearchCV(estimator = clf(), param_grid = clf_param_map[clf], cv=5, n_jobs=-1)


	logging.info("Creating data processing pipeline.............................................")

	pipeline  = Pipeline([('feature_selection',feature_sele),('clf_cv',grid_search)])
	

	for i in range(1,11):
		print(f"Beginning run {i} of 5-fold CV.......................................................................")
		logging.info(f"Beginning run {i} of 5-fold CV.......................................................................")
		
		
		grpkfold = StratifiedGroupKFold(n_splits=5)
		splits = grpkfold.split(data,labels,groups)
		for train_index, test_index in splits:
			data = np.asarray(data)
			labels = np.asarray(labels)
			train_data = data[train_index]
			test_data = data[test_index]
			train_labels = labels[train_index]
			test_labels = labels[test_index]


			logging.info("Fitting data processing pipeline.............................................")
		
			
			pipeline.fit_transform(train_data,train_labels)
			logging.info(f"Number of input features: {pipeline[1].n_features_in_}")
			n_feature_list.append(pipeline[1].n_features_in_)
			logging.info(pipeline[1].best_estimator_)

			logging.info("Going into cross validation  testing...............................")
			# cv_score = cross_val_predict(pipeline,data,labels,cv=5)
			cv_pred = pipeline.predict(test_data)
			accuracy = accuracy_score(cv_pred,test_labels )
			score = precision_recall_fscore_support(cv_pred, test_labels)
			print(f"Accuracy:{accuracy}")
			classification_scores.append(accuracy)
			overall_conf_mat += confusion_matrix(labels[test_index], cv_pred)
			score_matrix+=score
		
		


	logging.info(classification_scores)
	logging.info("Mean Score: %.3f		Std: %.3f  "%(np.mean(classification_scores),np.std(classification_scores)))
	logging.info(score_matrix)
	logging.info(overall_conf_mat)
	logging.info(n_feature_importance)
	logging.info(np.mean(n_feature_list))
	logging.info(n_feature_list)

	logging.info(f"Execution time: {time.time() - start_time}")

	return pipeline


if __name__ == "__main__":

	from sklearn.model_selection import GroupShuffleSplit
	from sklearn.preprocessing import RobustScaler
	import feature_extraction as fe

	import pickle

	# participant_num = sys.argv[1]
	participant_num = 6
	
	
	data,labels,groups = load_data(participant_num)
	logging.info("Running true label analysis................................................................................")
	logging.info(f"Participant Number: {participant_num}")


	# Assign group values for blocks to preserve during splitting and cross validation

	logging.info(f"Shape of data matrix: {np.shape(data)}")
	logging.info(f"Shape of labels matrix: {np.shape(labels)}")
	logging.info(f"Shape of groups matrix: {np.shape(groups)}")


	#reshape to normalize data, reshape to original form
	data = np.asarray(data)
	scaler = RobustScaler().fit(data.reshape(data.shape[0], -1))
	data = scaler.transform(data.reshape(data.shape[0], -1)).reshape(data.shape[0],data.shape[1],data.shape[2])


	classifiers = [GradientBoostingClassifier,SVC,KNeighborsClassifier,LinearDiscriminantAnalysis,RandomForestClassifier]

	
	features= {
               "erp_based":[fe.erp_features, fe.offsets_features],
               "regularity":[fe.compute_spectral_entropy, fe.compute_periodicity],
               "harmonics":[fe.fundamental_freq],
			   "oscillatory_bands":[fe.compute_psd]
        }
	

	clf = classifiers[3]


	filename = f'{participant_num}ResultsLog.log'

	logging.basicConfig(filename=filename, filemode='w', format = '%(message)s',level=logging.INFO)
	logging.info('Logging into file')
	
	
	FEATURE_TYPE = "" #refer to features dictionary for key values, possible values ("erp_based","regularity","harmonics","oscillatory_bands")
	


	if FEATURE_TYPE in features:
		extr_feat = np.array([])
		for func in features[FEATURE_TYPE]:		
			if not np.any(extr_feat):
				extr_feat = func(data)
			else:
				result = func(data)
				extr_feat = np.hstack((extr_feat, result))
	
	

	if 'extr_feat' in locals() and extr_feat.size > 0:
		with open(f'P{participant_num}_{FEATURE_TYPE}_features.pickle', 'wb') as f:
			pickle.dump(extr_feat, f)

	
	estimator = train_classification_withholdout(data,labels,groups,clf)
		

