### PCA using 2 subsets

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

def calculate_class_mean(x_class):
		x_class_mean = []
		for i in range(0,52):
			class_mean = calculate_mean_image(x_class[i],name='Class Mean',show='no')
			class_mean = np.squeeze(class_mean)
			x_class_mean.append(class_mean)
		
		return np.array(x_class_mean)		

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
	for i in range(0,len(x_class_mean)):
		title = str("Class Mean "+str(i+1))
		plt.subplot(rows,cols,index), plt.imshow(np.reshape(x_class_mean[i],(46,56)).T, cmap = 'gist_gray')
		plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
		index = index+1
	if show == 'yes':
		plt.show()
	if save == 'yes':
		plt.savefig('Global and Class Mean')
	plt.close()

def calculate_Sb(x_class_mean,global_mean,show):
	mi_m = np.subtract(x_class_mean,global_mean).T
	Sb = np.dot(mi_m,mi_m.T)
		
	if show == 'yes':
		print("Global Mean (m) has shape",global_mean.T.shape,':\n',global_mean.T,'\n')
		print("Class Mean (mi) Matrix has shape",x_class_mean.T.shape,':\n',x_class_mean.T,'\n')
		print("Class Mean - Global Mean (mi-m) Matrix has shape",mi_m.shape,':\n',mi_m,'\n')
		print("Sb has shape",Sb.shape,':\n',Sb,'\n')
	
	return Sb

def calculate_Sw(x_train,x_class,x_class_mean,show='yes'):
	
	x_class_mean = x_class_mean.astype(float)
	
	Sw = [0]
	for i in range(0,52):
		x_mi = np.subtract(x_class[i],x_class_mean[i]).T
		Ai = np.dot(x_mi,x_mi.T)
		Sw = Sw + Ai

	'''
	for i in range(0,52):
		x_mi = np.subtract(x_class[i],x_class_mean[i])
		
		if i == 0:
			X_mi = x_mi
		if i > 0:
			X_mi = np.vstack((X_mi,x_mi))
	
	Sw = np.dot(X_mi.T,X_mi)
	print(Sw.shape)
	print(matrix_rank(Sw))	
	'''
	'''
	x_class_mean = x_class_mean.astype(float)
	
	Sw = [0]
	X_mi = x_train
	X_mi = X_mi.astype(float)
	for i in range(0,52):
		x_class[i] = x_class[i].astype(float)
		x_mi = np.subtract(x_class[i],x_class_mean[i])

		start = i*8
		end = start+8

		X_mi[start:end] = x_mi
		
		Si = np.dot(x_mi.T,x_mi)
		Sw = Sw + Si
	'''
	
		
	return Sw

def calculate_Wlda(SwSb,show):
	def get_sorted_eigenvectors(covariance_matrix):
		eigenvalues, eigenvectors = eigh(covariance_matrix) 
		idx = eigenvalues.argsort()[::-1]
		eigenvectors = eigenvectors[:,idx]
		return eigenvectors
	
	Wlda = get_sorted_eigenvectors(SwSb)
	if show == 'yes':
		print("Wlda has shape",Wlda.shape,':\n',Wlda,'\n')
		
	return Wlda[0:51]

def calculate_Wpca(x_train,global_mean,show):

	def calculate_batch_covariance_matrix(x_train,average_training_face):
		train_size = len(x_train.squeeze())
		A_train = np.subtract(x_train.squeeze(),average_training_face)
		A_train = A_train.T
		S = np.dot(A_train.T,A_train)/train_size
		return A_train, S
	
	def get_sorted_eigenvectors(covariance_matrix):
		eigenvalues, eigenvectors = eigh(covariance_matrix) 
		idx = eigenvalues.argsort()[::-1]
		eigenvectors = eigenvectors[:,idx]
		return eigenvectors

	A_pca,S_pca = calculate_batch_covariance_matrix(x_train,global_mean)
	eigenvec_pca = get_sorted_eigenvectors(S_pca)
	Wpca = np.dot(A_pca,eigenvec_pca)

	if show == 'yes':
		print("Wpca has shape",Wpca.shape,':\n',Wpca,'\n')

	return Wpca

def calculate_Wopt(Wlda,Wpca,show):
	Wopt = np.dot(Wlda.T,Wpca.T)
	if show == 'yes':
		print("Wopt has shape",Wopt.shape,':\n',Wopt,'\n')
	return Wopt

def plot_top_M_fisherfaces(eigenvec,M,mode):
	if mode == 'show':
		cols = 10
		rows = M/cols
		index = 1
		font_size = 10
		plt.figure(figsize=(20,20))
		for i in range(0,M):
			plt.subplot(rows,cols,index),plt.imshow(np.reshape(eigenvec[:,i],(46,56)).T, cmap = 'gist_gray')
			face_title = str("M="+str(i+1))
			plt.title(face_title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])	# set_position([a,b]) adjust b for height
			index = index+1
		overall_title = str("Top "+str(M)+" Fisherfaces for High-Dim PCA")
		plt.suptitle(overall_title)
		
		plt.show()
		plt.close()
		
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
	eigenvalues, eigenvectors = eigh(covariance_matrix) 
	idx = eigenvalues.argsort()[::-1]
	eigenvectors = eigenvectors[:,idx]
	eigenvalues = eigenvalues[idx]
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

def calculate_mean_image(x_train,name,show):
	train_size = len(x_train)
	sum_of_training_faces = [0]
	for i in range(0,train_size):
		sum_of_training_faces = sum_of_training_faces + x_train[i]
	average_training_face = np.array(sum_of_training_faces/train_size)[np.newaxis]
	if show == 'yes':
		print(name,"Mean Face has shape",average_training_face.shape,":\n",average_training_face, "\n")
	
	return average_training_face	

def calculate_batch_parameters(x_train, x_test, show):

	def calculate_batch_covariance_matrix(x_train,x_test,average_training_face):
		train_size = len(x_train.squeeze())
		A_train = np.subtract(x_train,average_training_face).squeeze()
		A_train = A_train.T
		A_test = np.subtract(x_test,average_training_face).squeeze()
		A_test = A_test.T
		S = np.dot(A_train,A_train.T)/train_size
		return A_train, A_test, S

	def calculate_eigenvectors_eigenvalues(covariance_matrix, A_train):
		eigenvalues, eigenvectors = eigh(covariance_matrix) 
		idx = eigenvalues.argsort()[::-1]
		eigenvalues = eigenvalues[idx]
		V = eigenvectors[:,idx]								# V = matrix of low-dimension eigenvectors							
		_U = V
		#_U = np.dot(A_train,V)								# reconstructed U from low-dimension eigenvectors
		#_U = normalize(_U,axis=0,norm='l2')
		return V, _U

	show=show
	start_time = time.clock()
	combined_batch_mean = calculate_mean_image(x_train,name='Batch',show=show)
	print(x_train.shape)
	A_train, A_test, S_batch = calculate_batch_covariance_matrix(x_train,x_test,combined_batch_mean)
	V, _U = calculate_eigenvectors_eigenvalues(S_batch, A_train)
	batch_training_time = time.clock() - start_time
	
	if show == 'yes':
		print("Combined Batch Mean has shape",combined_batch_mean.shape,":\n",combined_batch_mean, "\n")
		print("A_train has shape",A_train.shape,":\n",A_train, "\n")
		print("LD Eigenvector, V has shape",V.shape,":\n",V, "\n")
		print("Derived Eigenvector, U has shape",_U.shape,":\n",_U, "\n")
		
	return A_train, A_test, _U, combined_batch_mean, batch_training_time

def calculate_combined_subset_parameters(x_subset1, x_subset2, y_subset1, y_subset2, show):
	
	def calculate_subset_covariance_matrix(x,u):
		N = len(x)
		A = np.subtract(x,u).T
		S = np.dot(A,A.T)/N
		return A,S
		
	def get_sorted_eigenvectors(covariance_matrix):
		eigenvalues, eigenvectors = eigh(covariance_matrix) 
		idx = eigenvalues.argsort()[::-1]
		eigenvectors = eigenvectors[:,idx]
		return eigenvectors
	
	show=show
	
	N1 = len(x_subset1)
	N2 = len(x_subset2)
	N3 = N1+N2
	u1 = calculate_mean_image(x_subset1,name='Subset1',show=show)
	u2 = calculate_mean_image(x_subset2,name='Subset2',show=show)
	N1u1 = np.multiply(N1,u1)
	N2u2 = np.multiply(N2,u2)
		
	u3 = (1/(N1+N2))*(N1u1 + N2u2)	### calculating combined mean face
		
	A1,S1 = calculate_subset_covariance_matrix(x_subset1,u1) 
	A2,S2 = calculate_subset_covariance_matrix(x_subset2,u2)
	
	N1overN3timesS1 = np.multiply((N1/N3),S1)
	N2overN3timesS2 = np.multiply((N2/N3),S2)
	u1_u2 = np.subtract(u1,u2)								### u1-u2
	N1N2overN1plusN2squared_u1u2 = (N1*N2)/((N1+N2)**2)*np.dot(u1_u2,u1_u2.T).squeeze()
	
	S3 = np.add(np.add(N1overN3timesS1,N2overN3timesS2),N1N2overN1plusN2squared_u1u2)	### combined covariance matrix
		
	U1 = get_sorted_eigenvectors(S1)	### subset1 covariance matrix 
	start_time = time.time()
	U2 = get_sorted_eigenvectors(S2)	### subset2 covariance matrix 
	
	P1 = U1[:,0:x_subset1.shape[0]]
	P2 = U2[:,0:x_subset2.shape[0]]

	horizontal_stack_matrix = np.hstack((P1,P2,u1_u2.T))	### stacks the matrix
	PHI,R = np.linalg.qr(horizontal_stack_matrix)
	S3_PHI = np.dot(S3,PHI)
	PHIT_S3_PHI = np.dot(PHI.T,S3_PHI)
	
	
	R = get_sorted_eigenvectors(PHIT_S3_PHI)
	P3 = np.dot(PHI,R)									
	training_time = time.time() - start_time
	x3 = np.vstack((x_subset1,x_subset2))					### combined subset of 1 and 2: face data
	y3 = y_subset1+y_subset2								### combined subset of 1 and 2: labels
	A3 = np.subtract(x3,u3).squeeze().T						### combined A of subset 1 and 2
	
	if show == 'yes':
		print("Combined Mean Face has shape",u3.shape,":\n",u3, "\n")
		print("Subset 1 Covariance Matrix has shape",S1.shape,":\n",S1, "\n")
		print("Subset 2 Covariance Matrix has shape",S2.shape,":\n",S2, "\n")
		print("Combined Covariance Matrix has shape",S3.shape,":\n",S3, "\n")
		print("P1 has shape",P1.shape,":\n",P1, "\n")
		print("P2 has shape",P2.shape,":\n",P2, "\n")
		print("PHI has shape",PHI.shape,":\n",PHI, "\n")
		print("PHIT_S3_PHI has shape",PHIT_S3_PHI.shape,":\n",PHIT_S3_PHI, "\n")
		print("P3 has shape",P3.shape,":\n",P3, "\n")
		
	return P3,x3,y3,A3,u3,S3,training_time	

def reconstruct_image_batch_PCA(X_train,Y_train,A,_U,sample_list,M_list,average_training_face,main_title,plot,save):
	
	def calculate_reconstruction_error(reconstructed_image,original_face):
		reconstructed_image = np.array([int(x) for x in reconstructed_image.squeeze()])
		reconstruction_error = np.subtract(original_face,reconstructed_image)
		reconstruction_error = norm(reconstruction_error)/norm(original_face)*100
		reconstruction_error_list.append(reconstruction_error)
		M_error_list.append(M)
		return reconstruction_error_list, M_error_list
	
	font_size = 10
	index = 1
	count = 1
	cols = len(M_list)
	rows = len(sample_list)
	plt.figure(figsize=(20,10))
	M_error_list = []
	reconstruction_error_list = []
	for sample in sample_list:
		original_face = X_train[sample]		
		original_face = np.array(original_face)[np.newaxis]
		plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(original_face,(46,56)).T, cmap = 'gist_gray')
		title = str("F"+str(Y_train[sample])+"-Original")
		plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
		index=index+1
		for M in M_list:
			__U = _U[:,0:M]
			#__U = _U[:,0:M]
			W = np.dot(A.T,__U)	
			reconstructed_image = average_training_face
			for i in range(0,M):
				print(main_title, ": F",Y_train[sample], "[", count, "/", rows, "] with M =", M, "... ( M = ", i+1, "/", M, ")    ", end="\r")
				au = W[sample,i]*__U[:,i]
				reconstructed_image = reconstructed_image + au
			if sample == sample_list[0]:
				reconstruction_error_list, M_error_list = calculate_reconstruction_error(reconstructed_image,original_face)
			title = str("F"+str(Y_train[sample])+" [M=" + str(M) + "]")
			plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(reconstructed_image,(46,56)).T, cmap = 'gist_gray')
			plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
			index = index+1
		count = count+1
	plt.suptitle(main_title)
	if save == 'yes':
		plt.savefig(main_title)
	if plot == 'yes':
		plt.show()
	plt.close()
	title = str(str(main_title)+" Reconstruction Error")
	reconstruction_error_df = DataFrame(M_error_list, columns = ['M'])	
	reconstruction_error_df[title] = reconstruction_error_list
	print("\n")
	return reconstruction_error_df
	
def reconstruct_image_incremental_PCA(X_train,Y_train,A,eigenvectors,sample_list,M_list,average_training_face,main_title,plot,save):
	
	def calculate_reconstruction_error(reconstructed_image,original_face):
		reconstructed_image = np.array([int(x) for x in reconstructed_image.squeeze()])
		reconstruction_error = np.subtract(original_face,reconstructed_image)
		reconstruction_error = norm(reconstruction_error)/norm(original_face)*100
		reconstruction_error_list.append(reconstruction_error)
		M_error_list.append(M)
		return reconstruction_error_list, M_error_list
	
	font_size = 8
	index = 1
	count = 1
	cols = len(M_list)
	rows = len(sample_list)
	plt.figure(figsize=(20,10))
	M_error_list = []
	reconstruction_error_list = []
	for sample in sample_list:	
		original_face = X_train[sample]		
		original_face = np.array(original_face)[np.newaxis]
		plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(original_face,(46,56)).T, cmap = 'gist_gray')
		title = str("F"+str(Y_train[sample])+"-Original")
		plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
		index=index+1
		for M in M_list:
			U = eigenvectors[:,0:M]
			W = np.dot(A.T,U)
			reconstructed_image = average_training_face
			for i in range(0,M):
				print(main_title, ": F",Y_train[sample], "[", count, "/", rows, "] with M =", M, "... ( M = ", i+1, "/", M, ")    ", end="\r")
				au = W[sample,i]*U[:,i]
				reconstructed_image = reconstructed_image + au
			if sample == sample_list[0]:
				reconstruction_error_list, M_error_list = calculate_reconstruction_error(reconstructed_image,original_face)
			title = str("F"+str(Y_train[sample])+" [M=" + str(M) + "]")
			plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(reconstructed_image,(46,56)).T, cmap = 'gist_gray')
			plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
			index = index+1
		count = count+1
	plt.suptitle(main_title)
	if save == 'yes':
		plt.savefig(main_title)
	if plot == 'yes':
		plt.show()
	plt.close()
	title = str(str(main_title)+" Reconstruction Error")
	reconstruction_error_df = DataFrame(M_error_list, columns = ['M'])	
	reconstruction_error_df[title] = reconstruction_error_list
	print("\n")
	return reconstruction_error_df

def plot_df(df1, df2, plot_title, x_axis, y_axis, plot,save):									# function for plotting high-dimension and low-dimension eigenvalues
	y1 = df1[df1.columns[1]]
	x1 = df1['M']
	y2 = df2[df2.columns[1]]
	x2 = df2['M']

	plt.figure(figsize=(8,8))
	plt.plot(x1, y1, color = 'red')									
	plt.plot(x2, y2, color = 'blue')									
	plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)				# parameters for plot grid
	plt.xticks(np.arange(0,max(x1)*1.1, int(max(x1)/10)))								# adjusting the intervals to 250
	plt.yticks(np.arange(0,max(y1)*1.1, int(max(y1)/10)))
	plt.title(plot_title).set_position([0.5,1.05])
	plt.xlabel(x_axis)
	plt.ylabel(y_axis)			
	plt.legend(loc = 'best')											# creating legend and placing in at the top right
	if save == 'yes':
		plt.savefig(plot_title)
	if plot == 'yes':
		plt.show()
	plt.close()

def Batch_NN_classifier(M_list_NN,A_train,A_test,X_test,Y_train,Y_test,_U,average_training_face,main_title,show):
	test_size = len(Y_test)
	train_size = len(Y_train)
	success_rate_list = []
	success_rate_M_list = []
	for M in M_list_NN:
		success = 0
		__U = normalize(_U,axis=0,norm='l2')
		__U = np.multiply(__U,-1)
		__U = __U[:,0:M]
		W_train = np.dot(A_train.T,__U)	
		W_test = np.dot(A_test.T,__U)
		for ntest in range(0,test_size):
			minimum_error = 999999999
			start_time = time.time()
			
			for ntrain in range(0,train_size):
				error =  W_test[ntest]-W_train[ntrain]
				l2_error = norm(error)
				
				if l2_error<minimum_error:
					minimum_error = l2_error
					label = Y_train[ntrain]
					pos = ntrain
					
			if Y_test[ntest] == label:
				success = success + 1

		success_rate = success/test_size*100
		success_rate_list.append(success_rate)
		success_rate_M_list.append(M)
		if show == 'yes':
			print("For M =",M,"success rate is",success_rate,"%")

	df = DataFrame(success_rate_M_list, columns = ['M'])
	title = str(main_title+' Success Rate')
	df[title] = success_rate_list
	x = df['M']
	y = df[title]
	return df

def Subset_NN_classifier(M_list_NN,A_train,A_test,X_test,Y_train,Y_test,eigenvectors_hd,average_training_face,main_title,show):
	test_size = len(Y_test)
	train_size = len(Y_train)
	success_rate_list = []
	success_rate_M_list = []
	for M in M_list_NN:
		success = 0
		U = eigenvectors_hd[:,0:M]
		W_train = np.dot(A_train.T,U)	
		W_test = np.dot(A_test.T,U)
		for ntest in range(0,test_size):
			minimum_error = 999999999
			start_time = time.time()
			for ntrain in range(0,train_size):
				error =  W_test[ntest]-W_train[ntrain]
				l2_error = norm(error)
				if l2_error<minimum_error:
					minimum_error = l2_error
					label = Y_train[ntrain]
					pos = ntrain
			if Y_test[ntest] == label:
				success = success + 1

		success_rate = success/test_size*100
		success_rate_list.append(success_rate)
		success_rate_M_list.append(M)
		if show == 'yes':
			print("For M =",M,"success rate is",success_rate,"%")
			
	df = DataFrame(success_rate_M_list, columns = ['M'])
	title = str(main_title+' Success Rate')
	df[title] = success_rate_list
	x = df['M']
	y = df[title]
	return df

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
	global_mean = global_mean.squeeze()
	
	#### CALCULATE CLASS MEAN
	x_class_mean = calculate_class_mean(x_class)
	
	#### PRINT AND SAVE ALL THE MEAN FACES (CLASS AND GLOBAL MEANS)
	print_save_all_mean_faces(x_class_mean,global_mean,show='no',save='no')		### Only choose save or show

	#### COMPUTE Sb = (mi-m)(mi-m).T
	Sb = calculate_Sb(x_class_mean,global_mean,show='no')	
	
	#### COMPUTE Sw = S1+S2+...+S52
	Sw = calculate_Sw(X_train,x_class,x_class_mean,show='no')
	
	#### COMPUTE Wpca
	Mpca_list = list(range(100, matrix_rank(Sw), 20))
	Mpca_list.append(matrix_rank(Sw))
	#Mpca_list=[100]
	
	Mpca = 364
	Wpca = calculate_Wpca(X_train,global_mean,show='no')
	Wpca = Wpca[:,0:Mpca]
	print("Wpca =",Wpca.shape)
	#print("Sb =",Sb.shape)
	#print("Sw =",Sw.shape,'\n')
		
	#### COMPUTE Wpca_T.Sb.Wpca --- E1
	WpcaT_Sb_Wpca = np.dot(Wpca.T,np.dot(Sb,Wpca))	
	E1 = WpcaT_Sb_Wpca
	#print("E1 =",E1.shape)
		
	#### COMPUTE Wpca_T.Sw.Wpca --- E2
	WpcaT_Sw_Wpca = np.dot(Wpca.T,np.dot(Sw,Wpca))	
	E2 = WpcaT_Sw_Wpca
	#print("E2 =",E2.shape)
		
	#### COMPUTE inv(E2).E1
	Slda = np.dot(inv(E2),E1)
	eigenvalues,Wlda = get_sorted_eigenvalues_eigenvectors(Slda)
	print(Wlda.shape)
	Wlda = Wlda[:,0:51]
	print("Wlda =",Wlda.shape,'\n')
		
	#### COMPUTE Wopt
	Wopt = np.dot(Wlda.T,Wpca.T)
	Wopt = Wopt.T
	Wopt = normalize(Wopt,axis=1,norm='l2')
	print("Wopt =",Wopt.shape)
		
	#### PLOT FISHERFACES
	plot_top_M_fisherfaces(Wopt,M=50,mode='show')
	
	#### COMPUTE AND PLOT PROJECTED FACES
	plot_projected_3faces_onto_fisherspace(Wopt,X_train,global_mean,mode='show')
	#### COMPUTE RECOGNITION RATE
	sys.exit()	
	def PCA_LDA_classifier(Mpcalda_list,X_train,X_test,Y_train,Y_test,Wopt,main_title,show):
		test_size = len(Y_test)
		train_size = len(Y_train)
		success_rate_list = []
		success_rate_M_list = []
		for Mpcalda in Mpcalda_list:
			success = 0
			Wpcalda = Wopt
			#print(X_train.T.shape)
			#print(Wpcalda.shape)
			projected_train = np.dot(Wpcalda.T,X_train.T)
			projected_train = projected_train.T
				
			projected_test = np.dot(Wpcalda.T,X_test.T)
			projected_test = projected_test.T
				
			#print(projected_train.shape)
			#print(projected_test.shape)

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
			success_rate_list.append(success_rate)
			success_rate_M_list.append(Mpcalda)
			#if show == 'yes':
				#print("For Mpcalda =",Mpcalda,"success rate is",success_rate,"%")
		print("For Mpcalda =",Mpcalda,"success rate is",success_rate,"%\n")	
		df = DataFrame(success_rate_M_list, columns = ['M'])
		title = str(main_title+' Success Rate')
		df[title] = success_rate_list
		x = df['M']
		y = df[title]
		return df

	Mpcalda_list = list(range(40, 300, 20))
	Mpcalda_list.append(matrix_rank(Sb))
	Mpcalda_list = [100]
	#print(Mpca_list)
	#print(Mpcalda_list)
		
	PCA_LDA_classifier(Mpcalda_list,X_train,X_test,Y_train,Y_test,Wopt,main_title='PCA-LDA Classification',show='yes')
		
	sys.exit()
	
	return 0
	
main()
