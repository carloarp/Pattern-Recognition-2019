### PCA-LDA Ensemble
### Varies T while keeping M0, M1 constant

############################### IMPORT DEPENDENCIES ######################################################

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import cv2
import sys
import time
import os
import psutil
import random
import pandas as pd

from scipy.io import loadmat
from sklearn.decomposition import PCA
from scipy.linalg import eigh
from scipy.linalg import eig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import matrix_rank
from numpy.linalg import inv
from scipy.linalg import orth
from statistics import mode, StatisticsError

										# comment out this code when running on jupyter
dir = os.getcwd()						# gets the working directory of the file this python script is in
os.chdir (dir)							# changes the working directory to the file this python script is in
										# also change plt.savefig to plt.show

############################### FUNCTIONS START HERE ######################################################
def split_data_into_classes(x_train,y_train):
		x_class = []
		y_class = []
		
		for i in range(0,52):
			i = i*8
			y_class.append(y_train[i:i+8])
			x_class.append(x_train[i:i+8])

		return x_class, y_class

def calculate_class_mean(x_class,show):
		x_class_mean = []
		for i in range(0,52):
			class_mean = calculate_mean_image(x_class[i],name='Class Mean',show='no')
			class_mean = np.squeeze(class_mean)
			x_class_mean.append(class_mean)
		x_class_mean = np.array(x_class_mean).T
		if show=='yes':
			describe(x_class_mean,'Class Mean')
		return x_class_mean	

def calculate_mean_image(x_train,name,show):
	train_size = len(x_train)
	sum_of_training_faces = [0]
	for i in range(0,train_size):
		sum_of_training_faces = sum_of_training_faces + x_train[i]
	average_training_face = np.array(sum_of_training_faces/train_size)[np.newaxis]
	average_training_face = average_training_face.T
	if show == 'yes':
		describe(average_training_face,name)
	return average_training_face	
			
def print_save_all_mean_faces(x_class_mean,global_mean,show,save):
	rows = 4
	cols = 14
	index = 1
	font_size = 10
	plt.figure(figsize=(20,10))
	plt.subplot(rows,cols,index), plt.imshow(np.reshape(global_mean,(46,56)).T, cmap = 'gist_gray')
	title = str("Global Mean")
	plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
	index=index+1
	for i in range(0,x_class_mean.shape[1]):
		title = str("Class Mean "+str(i+1))
		plt.subplot(rows,cols,index), plt.imshow(np.reshape(x_class_mean[:,i],(46,56)).T, cmap = 'gist_gray')
		plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
		index = index+1
	if show == 'yes':
		plt.show()
	if save == 'yes':
		plt.savefig('Global and Class Mean')
	plt.close()

def calculate_Sb(x_class_mean,global_mean,show):
	mi_m = np.subtract(x_class_mean,global_mean)
	Sb = np.dot(mi_m,mi_m.T)	
	if show == 'yes':
		describe(Sb,'Sb')
	return Sb

def calculate_Sw(x_train,x_class,x_class_mean,show):
	x_class_mean = x_class_mean.astype(float)
	Sw = [0]
	for i in range(0,52):
		x_class_mean_i = np.array(x_class_mean[:,i])[np.newaxis].T
		x_mi = np.subtract(x_class[i].T,x_class_mean_i)
		Ai = np.dot(x_mi,x_mi.T)
		Sw = Sw + Ai
	if show=='yes':
		describe(Sw,'Sw')
	return Sw

def calculate_Wpca(x_train,global_mean,show):

	def calculate_batch_covariance_matrix(x_train,average_training_face):
		train_size = len(x_train.squeeze())
		A_train = np.subtract(x_train.squeeze().T,average_training_face)
		S = np.dot(A_train.T,A_train)/train_size
		return A_train, S
	
	def get_sorted_eigenvectors(covariance_matrix):
		eigenvalues, eigenvectors = eig(covariance_matrix) 
		idx = eigenvalues.argsort()[::-1]
		eigenvectors = eigenvectors[:,idx]
		eigenvectors = eigenvectors.real
		return eigenvectors

	A_pca,S_pca = calculate_batch_covariance_matrix(x_train,global_mean)
	eigenvec_pca = get_sorted_eigenvectors(S_pca)
	Wpca = np.dot(A_pca,eigenvec_pca)

	if show == 'yes':
		describe(Wpca,'Wpca')

	return Wpca

def plot_projected_3faces_onto_fisherspace(Wopt,X_train,average_training_face,mode):
	if mode == 'show':
		projected = np.dot(Wopt.T,X_train.T)
		print("Projected =",projected.shape)
		projected = projected.T
		
		c = ['red','blue','green','black']
		m = ['o','x','v','^']
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		count = 0
			
		for i in range(0,24):
			x = projected[i][0]
			y = projected[i][1]
			z = projected[i][2]
			
			if (i%8) == 0:
					count = count+1
			ax.scatter(x, y, z, c=c[count], marker=m[count])
			
		plt.show()
		plt.close()
		
def get_sorted_eigenvalues_eigenvectors(covariance_matrix):
	eigenvalues, eigenvectors = eig(covariance_matrix) 
	idx = eigenvalues.argsort()[::-1]
	eigenvectors = eigenvectors[:,idx]
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors.real
	return eigenvalues,eigenvectors
	
def print_image(face_image):			# function for plotting an image
	face_image = np.reshape(face_image, (46,56))
	face_image = face_image.T
	plt.imshow(face_image, cmap='gist_gray')
	plt.show()

def save_image(face_image,title):		# function for saving an image
	face_image = np.reshape(face_image, (46,56))
	face_image = face_image.T
	plt.imshow(face_image, cmap='gist_gray')
	plt.title(title)
	plt.savefig(title)
	
def partition_data(face_data,face_labels,split,show):
	number_of_faces = face_data.shape[1]
	number_of_pixels = face_data.shape[0]
	face_labels_array = []
	face_pixels_array = []

	# Create an array that only contains the column pixels of the image
	for i in range (0,number_of_faces):
		face_labels_array.append(face_labels[0,i])					# Array that contains the labels
		face_pixels_array.append(face_data[:,i])					# Array that contrains the face_data
	x_train = []
	x_test = []
	y_train = []
	y_test = []
	for i in range(0,52):
		i = i*10
		face_array = face_pixels_array[i:i+10]
		labels_array = face_labels_array[i:i+10]
		face_train, face_test, label_train, label_test = train_test_split(face_array, labels_array, test_size=split, random_state=42)
		x_train.extend(face_train)
		y_train.extend(label_train)
		x_test.extend(face_test)
		y_test.extend(label_test)
	if show == 'yes':
		print("\n")
		print("Split = ", split, " ---> Train =", 1-split, ", Test =", split)
		print("Size of Train Data (Pixel Vectors): ", 	len(x_train))
		print("Size of Train Data (Labels)", 			len(y_train))
		print("Size of Test Data (Pixel Vectors): ", 	len(x_test))
		print("Size of Test Data (Labels)", 			len(y_test), "\n")
	train_size = len(x_train)
	test_size = len(x_test)
	x_train = np.array(x_train)[np.newaxis]
	x_test = np.array(x_test)[np.newaxis]
	
	return x_train, y_train, x_test, y_test, train_size, test_size

def plot_df(df1,plot_title,x_axis,y_axis,plot,save):						# function for plotting high-dimension and low-dimension eigenvalues
	y1 = df1[df1.columns[1]]
	x1 = df1[df1.columns[0]]
	plt.figure(figsize=(8,8))
	plt.plot(x1, y1, color = 'red')																
	plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)				# parameters for plot grid
	plt.xticks(np.arange(0,max(x1)*1.1, int(max(x1)/10)))					# adjusting the intervals to 250
	plt.yticks(np.arange(0,max(y1)*1.1, int(max(y1)/10)))
	plt.title(plot_title).set_position([0.5,1.05])
	plt.xlabel(x_axis)
	plt.ylabel(y_axis)			
	plt.legend(loc = 'best')												# creating legend and placing in at the top right
	if save == 'yes':
		plt.savefig(plot_title)
	if plot == 'yes':
		plt.show()
	plt.close()
	
def describe(matrix,name):
	rank = matrix_rank(matrix)
	print(name,"has shape",matrix.shape,"and rank",rank,":\n",matrix,"\n")

def PCA_LDA_classifier(X_train,X_test,Y_train,Y_test,Wopt):
	test_size = len(Y_test)
	train_size = len(Y_train)

	success = 0
	success_rate = 0
	Wpcalda = Wopt

	projected_train = np.dot(Wpcalda.T,X_train.T)
	projected_train = projected_train.T
					
	projected_test = np.dot(Wpcalda.T,X_test.T)
	projected_test = projected_test.T

	for ntest in range(0,test_size):
		minimum_error = 99999999999999999999999999
		start_time = time.time()
		for ntrain in range(0,train_size):
			error =  projected_test[ntest]-projected_train[ntrain]
			l2_error = norm(error)
			if l2_error<minimum_error:
				minimum_error = l2_error
				label = Y_train[ntrain]
				pos = ntrain
		if Y_test[ntest] == label:
			success = success + 1

	success_rate = success/test_size*100

	return success_rate

def most_common(l):
	try:
		return mode(l)
	except StatisticsError as e:
		# will only return the first element if no unique mode found
		if 'no unique mode' in e.args[0]:
			return l[0]
		# this is for "StatisticsError: no mode for empty data"
		# after calling mode([])
		raise	

def plot_3df(df,plot_title,x_axis,y_axis,plot,save):						# function for plotting high-dimension and low-dimension eigenvalues
	x1 = df[df.columns[0]]
	y1 = df[df.columns[1]]
	y2 = df[df.columns[2]]
	y3 = df[df.columns[3]]
	min1 = df[df.columns[1]].min()
	min2 = df[df.columns[2]].min()
	min3 = df[df.columns[3]].min()
	df_min = min(min1,min2,min3)
	plt.figure(figsize=(8,8))
	plt.plot(x1, y1, color = 'red')
	plt.plot(x1, y2, color = 'blue')
	plt.plot(x1, y3, color = 'green')
	plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)				# parameters for plot grid
	plt.xticks(np.arange(0,max(x1)*1.1, int(max(x1)/10)))					# adjusting the intervals to 250
	plt.yticks(np.arange(df_min*0.9,100, int(max(y1)/10)))
	plt.title(plot_title).set_position([0.5,1.05])
	plt.xlabel(x_axis)
	plt.ylabel(y_axis)			
	plt.legend(loc = 'best')												# creating legend and placing in at the top right
	if save == 'yes':
		plt.savefig(plot_title)
	if plot == 'yes':
		plt.show()
	plt.close()
	
		
######################################### MAIN STARTS HERE ###########################################################	


def main():	
	#### LOAD FACE DATA
	incremental_PCA_training_time_list = []
	batch_PCA_training_time_list = []
	combined_df = pd.DataFrame(columns=[])
	mat_content = loadmat('face(1).mat')			# unpacks the .mat file
	print(" ")
	face_data = mat_content['X']
	face_labels = mat_content['l']	

	#### PARTITION DATA INTO TRAIN AND TEST SET
	x_train, Y_train, x_test, Y_test, train_size, test_size = partition_data(face_data,face_labels,split=0.2,show='no')
	X_train = x_train.squeeze()				# This is the matrix that contains all of the face training data
	X_test = x_test.squeeze()				

	#### SPLIT TRAINING DATA INTO 52 CLASSES
	x_class, y_class = split_data_into_classes(X_train,Y_train)					### x_class[class][index]
	
	#### CALCULATE GLOBAL MEAN
	global_mean = calculate_mean_image(X_train,name='Global Mean',show='no')

	#### CALCULATE CLASS MEAN
	x_class_mean = calculate_class_mean(x_class,show='no')
	
	#### PRINT AND SAVE ALL THE MEAN FACES (CLASS AND GLOBAL MEANS)
	print_save_all_mean_faces(x_class_mean,global_mean,show='no',save='no')		### Only choose save or show

	#### COMPUTE Sb = (mi-m)(mi-m).T
	Sb = calculate_Sb(x_class_mean,global_mean,show='no')	
	
	#### COMPUTE Sw = S1+S2+...+S52
	Sw = calculate_Sw(X_train,x_class,x_class_mean,show='no')
	
	#### COMPUTE Wpca
	W_pca = calculate_Wpca(X_train,global_mean,show='no')
	W_pca = W_pca[:,0:matrix_rank(W_pca)]
	
	#### GENERATE 'T' RANDOM SUBSPACES
	np.random.seed(42)
	M0 = 50
	M1 = 100
	Wpca_M0 = W_pca[:,0:M0]

	majority_result_list = []
	average_accuracy_list = []
	sum_result_list = []
	T_list = []
	
	for T in range(5,55,5):
	#for T in range(5,15,5):
	
		Wopt_list = []
		remaining_index = np.arange(415)[M0:]
		
		for t in range(0,T):
			np.random.shuffle(remaining_index)
			random_remaining_index = remaining_index[:M1]
			
			### GENERATE M1 RANDOM EIGENFACES
			Wpca_M1 = W_pca[:,random_remaining_index]
			
			### Mpca = M0 + M1
			Wpca = np.hstack((Wpca_M0,Wpca_M1))

			#### COMPUTE Wpca_T.Sb.Wpca --- E1
			WpcaT_Sb_Wpca = np.dot(Wpca.T,np.dot(Sb,Wpca))	
			E1 = WpcaT_Sb_Wpca

			#### COMPUTE Wpca_T.Sw.Wpca --- E2
			WpcaT_Sw_Wpca = np.dot(Wpca.T,np.dot(Sw,Wpca))	
			E2 = WpcaT_Sw_Wpca
			
			#### COMPUTE inv(E2).E1 --- Wlda
			Slda = np.dot(inv(E2),E1)
			eigenvalues,Wlda = get_sorted_eigenvalues_eigenvectors(Slda)

			#### COMPUTE Wopt
			Wopt = np.dot(Wlda.T,Wpca.T)
			Wopt = Wopt.T
			Wopt = normalize(Wopt,axis=1,norm='l2')
			Wopt_list.append(Wopt)
		
		#### MAJORITY VOTE RECOGNITION 
		test_size = len(Y_test)
		train_size = len(Y_train)
		individual_success = 0
		success = 0
		success_rate = 0
		
		for ntest in range(0,test_size):
			result_list = []
			
			for t in range(0,T):
				minimum_error = 99999999999999999999999999
				Wpcalda = Wopt_list[t]
				projected_train = np.dot(Wpcalda.T,X_train.T)
				projected_train = projected_train.T
				projected_test = np.dot(Wpcalda.T,X_test.T)
				projected_test = projected_test.T

				for ntrain in range(0,train_size):
					error =  projected_test[ntest]-projected_train[ntrain]
					l2_error = norm(error)
						
					if l2_error<minimum_error:
						minimum_error = l2_error
						label = Y_train[ntrain]
						pos = ntrain
				
				if Y_test[ntest] == label:
					individual_success = individual_success + 1
				
				result_list.append(label)	
				print("T =",T,", Test data =",ntest+1,"/",test_size,", Classifier =",t+1,",Prediction =",label,end="\r")			
			majority = most_common(result_list)
			if Y_test[ntest] == majority:
				success = success + 1
		
		individual_success_rate = (individual_success/test_size)*100/T
		success_rate = success/test_size*100
		print("\n")
		print("Majority Success Rate for T =",T," is ",success_rate,"%")
		print("Average Model Success Rate for T =",T," is ",individual_success_rate,"%")
		
		#### SUM RECOGNITION 
		Wpcalda_sum = [0]
		for t in range(0,T):
			Wpcalda_sum = Wpcalda_sum + Wopt_list[t]
		sum_success_rate = PCA_LDA_classifier(X_train,X_test,Y_train,Y_test,Wpcalda_sum)
		print("Sum Success Rate for T =",T," is ",sum_success_rate,"%\n")
		
		majority_result_list.append(success_rate)
		average_accuracy_list.append(individual_success_rate)
		sum_result_list.append(sum_success_rate)
		T_list.append(T)
	
	ensemble_df = pd.DataFrame(columns=['T','Majority Success Rate','Average Success Rate','Sum Success Rate'])
	ensemble_df['T'] = T_list
	ensemble_df['Majority Success Rate'] = majority_result_list
	ensemble_df['Average Success Rate'] = average_accuracy_list
	ensemble_df['Sum Success Rate'] = sum_result_list
	
	print(ensemble_df)
	plot_3df(ensemble_df,plot_title='T against different Success Rates',x_axis='T',y_axis='Success Rate (%)',plot='yes',save='no')
	
	return 0
	
main()
