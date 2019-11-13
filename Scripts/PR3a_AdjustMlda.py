### PCA-LDA
### Varies Mlda while keeping Mpca fixed
### Plots recognition rate

############################### IMPORT DEPENDENCIES ######################################################

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import cv2
import sys
import time
import os
import psutil
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
	Wpca = calculate_Wpca(X_train,global_mean,show='no')
	
	list1 = list(range(1, 10, 1))
	list2 = list(range(10, matrix_rank(Sb), 5))
	list2.append(matrix_rank(Sb))
	Mlda_list = list1+list2
	success_rate_list = []
	
	df = pd.DataFrame(columns=['Mlda','Success Rate'])

	Mpca = 364 
	Wpca = Wpca[:,0:Mpca]
		
	#### COMPUTE Wpca_T.Sb.Wpca --- E1
	WpcaT_Sb_Wpca = np.dot(Wpca.T,np.dot(Sb,Wpca))	
	E1 = WpcaT_Sb_Wpca

	#### COMPUTE Wpca_T.Sw.Wpca --- E2
	WpcaT_Sw_Wpca = np.dot(Wpca.T,np.dot(Sw,Wpca))	
	E2 = WpcaT_Sw_Wpca
	
	#### COMPUTE inv(E2).E1
	Slda = np.dot(inv(E2),E1)
	eigenvalues,W_lda = get_sorted_eigenvalues_eigenvectors(Slda)
	
	#### COMPUTE RECOGNITION RATE WHILE VARYING Mlda
	for Mlda in Mlda_list:
		Wlda = W_lda[:,0:Mlda]
			
		#### COMPUTE Wopt
		Wopt = np.dot(Wlda.T,Wpca.T)
		Wopt = Wopt.T
		Wopt = normalize(Wopt,axis=1,norm='l2')

		#### COMPUTE RECOGNITION RATE
		success_rate = PCA_LDA_classifier(X_train,X_test,Y_train,Y_test,Wopt)
		success_rate_list.append(success_rate)
		print("For Mpca =",Mpca,"and Mlda =",Mlda,", success rate is",success_rate,"%")	
	
	df['Success Rate'] = success_rate_list
	df['Mlda'] = Mlda_list
	plot_df(df,plot_title='Recognition Rate while varying Mlda',x_axis='Mlda',y_axis='Recognition Rate%',plot='yes',save='yes')	
	
	return 0
	
main()
