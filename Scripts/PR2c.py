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
from scipy.linalg import orth


										# comment out this code when running on jupyter
dir = os.getcwd()						# gets the working directory of the file this python script is in
os.chdir (dir)							# changes the working directory to the file this python script is in
										# also change plt.savefig to plt.show

############################### FUNCTIONS START HERE ######################################################
def calculate_combined_covariance(x_subset1,x_subset2,subset1_mean_face,subset2_mean_face,mode):
	mode = mode

	S1_hd, S1_ld = calculate_subset_covariance_matrix(x_subset1,subset1_mean_face,name='Subset1',mode=mode) # mode = 'print'/'none'
	S2_hd, S2_ld = calculate_subset_covariance_matrix(x_subset2,subset2_mean_face,name='Subset2',mode=mode) # mode = 'print'/'none'
		
	N_S1 = np.multiply((len(x_subset1)/(len(x_subset1)+len(x_subset2))),S1_hd)
	N_S2 = np.multiply((len(x_subset2)/(len(x_subset1)+len(x_subset2))),S2_hd)
		
	u1_u2 = np.subtract(subset1_mean_face,subset2_mean_face)
	N1N2_u1u2 = ((len(x_subset1)*len(x_subset2))/((len(x_subset1)+len(x_subset2))**2))*np.dot(u1_u2,u1_u2.T).squeeze()
		
	N_S1_N_S2 = np.add(N_S1,N_S2)
	S_combined = np.add(N_S1_N_S2,N1N2_u1u2)
	#S_combined = np.divide(S_combined,(len(x_subset1)+len(x_subset2)))
	
	rank_S_combined = matrix_rank(S_combined)
	if mode == 'print':
		print("S_combined has shape",S_combined.shape,"and rank",rank_S_combined,":\n",S_combined, "\n")
		
	return S_combined
		
def divide_training_data_into_4_subsets(x_train, y_train, train_size,mode):
	x_train = x_train.squeeze()
	x_train = list(x_train)
	x_subset1 = []
	x_subset2 = []
	x_subset3 = []
	x_subset4 = []
	y_subset1 = []
	y_subset2 = []
	y_subset3 = []
	y_subset4 = []
	x_subsets = [x_subset1, x_subset2, x_subset3, x_subset4]
	y_subsets = [y_subset1, y_subset2, y_subset3, y_subset4]
	for i in range(0,52):
		i = i*8
		x_train_remaining = x_train[i:i+8]
		y_train_remaining = y_train[i:i+8]
		split = 0.25

		for j in range(0,3):
			#print(j,"(before):",x_train_remaining)
			split = 2/len(y_train_remaining)
			x_train_remaining, x_subset, y_train_remaining, y_subset = train_test_split(x_train_remaining, y_train_remaining, test_size=split, random_state=42)
			x_subsets[j].extend(x_subset)
			y_subsets[j].extend(y_subset)
			#print(j,"(after) :",x_train_remaining,"\n")
			if j == 2:
				x_subsets[j+1].extend(x_train_remaining)
				y_subsets[j+1].extend(y_train_remaining)
	if mode == 'print':
		for k in range(0,4):
			print("Subset",k+1,"(face data) has length:",len(x_subsets[k]))
			print("Subset",k+1,"(label) has length:",len(y_subsets[k]),"\n")
					
	return np.array(x_subsets), y_subsets

def calculate_combined_mean(subset_list,subset_mean_list,mode):
		Nu_sum = [0]
		N_sum = 0
		for i in range(0,len(subset_list)):
			Nu = np.multiply(len(subset_list[i]),subset_mean_list[i])
			Nu_sum = np.add(Nu_sum,Nu)
			N_sum = N_sum + len(subset_list[i])
		combined_subset_mean = np.divide(Nu_sum,N_sum)
		if mode == 'print':
			print("Combined Subset Mean for",len(subset_list),"subsets has shape",combined_subset_mean.shape,":\n",combined_subset_mean, "\n")
		
		return combined_subset_mean	

def calculate_equivalent_batch_mean(subset_list,mode):
		mode=mode
		x_batch = []
		for subset in subset_list:
			for i in range(0,len(subset)):
				x_batch.append(subset[i])
		x_batch = np.array(x_batch)
		batch_mean = calculate_mean_image(x_batch,name='Batch',mode=mode)
	
		return x_batch, batch_mean
		
def calculate_non_zero_eigenvalues(eigenvalues,name):
	count_non_zero = 0
	count_actual_non_zero = 0
	count_negative = 0
	for i in range(0,len(eigenvalues)):
		if eigenvalues[i]>000000000.1:
			count_non_zero = count_non_zero+1
		if eigenvalues[i]>0:
			count_actual_non_zero = count_actual_non_zero + 1
		if eigenvalues[i]<0:
			count_negative = count_negative + 1
	print("Number of", name, "> 1e-9 =",count_non_zero)
	print("Number of", name, "> 0 =",count_actual_non_zero)
	print("Number of", name, "< 0 =",count_negative,"\n")

def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem
			
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
	
def partition_data(face_data,face_labels,split,mode):
	number_of_faces = face_data.shape[1]
	number_of_pixels = face_data.shape[0]
	face_labels_array = []
	face_pixels_array = []

	# Create an array that only contains the column pixels of the image
	for i in range (0,number_of_faces):
		face_labels_array.append(face_labels[0,i])					# Array that contains the labels
		face_pixels_array.append(face_data[:,i])					# Array that contrains the face_data
		#print(face_labels_array[i], " = ", face_pixels_array[i])
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
	if mode == 'print':
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

def calculate_mean_image(x_train,name,mode):
	train_size = len(x_train)
	sum_of_training_faces = [0]
	for i in range(0,train_size):
		sum_of_training_faces = sum_of_training_faces + x_train[i]
	average_training_face = np.array(sum_of_training_faces/train_size)[np.newaxis]
	if mode == 'print':
		print(name,"Mean Face has shape",average_training_face.shape,":\n",average_training_face, "\n")
	
	return average_training_face	

def calculate_covariance_matrix(x_train,x_test,average_training_face,mode):
	train_size = len(x_train.squeeze())
	A_train = np.subtract(x_train,average_training_face).squeeze()
	A_train = A_train.T
	A_test = np.subtract(x_test,average_training_face).squeeze()
	A_test = A_test.T
	rank_A_train = matrix_rank(A_train)
	rank_A_test = matrix_rank(A_test)
	start_time = time.time()
	high_dimension_covariance_matrix = np.dot(A_train,A_train.T)/train_size
	high_dimension_covariance_matrix_time = time.time() - start_time
	start_time = time.time()
	low_dimension_covariance_matrix = np.dot(A_train.T,A_train)/train_size
	low_dimension_covariance_matrix_time = time.time() - start_time
	rank_S_hd = matrix_rank(high_dimension_covariance_matrix)
	rank_S_ld = matrix_rank(low_dimension_covariance_matrix)
	if mode == 'print':
		print("A_train has shape",A_train.shape,"and rank",rank_A_train,":\n",A_train, "\n")
		print("A_test has shape",A_test.shape,"and rank",rank_A_test,":\n",A_test, "\n")
		print("High-Dimension Covariance Matrix [",np.round(high_dimension_covariance_matrix_time,3), "s ] has shape",high_dimension_covariance_matrix.shape,"and rank",rank_S_hd,":\n",high_dimension_covariance_matrix, "\n")
		print("Low-Dimension Covariance Matrix [",np.round(low_dimension_covariance_matrix_time,3), "s ] has shape",low_dimension_covariance_matrix.shape,"and rank",rank_S_ld,":\n",low_dimension_covariance_matrix, "\n")

	return A_train, A_test, high_dimension_covariance_matrix, low_dimension_covariance_matrix, high_dimension_covariance_matrix_time, low_dimension_covariance_matrix_time

def calculate_subset_covariance_matrix(x,average_training_face,name,mode):
	train_size = len(x)
	A = np.subtract(x,average_training_face)
	A = A.T
	rank_A = matrix_rank(A)
	start_time = time.time()
	S_hd = np.dot(A,A.T)/train_size
	S_hd_time = time.time() - start_time
	start_time = time.time()
	S_ld = np.dot(A.T,A)/train_size
	S_ld_time = time.time() - start_time
	rank_S_hd = matrix_rank(S_hd)
	rank_S_ld = matrix_rank(S_ld)
	if mode == 'print':
		print(name,": A has shape",A.shape,"and rank",rank_A,":\n",A, "\n")
		print(name," : High-Dimension Covariance Matrix [",np.round(S_hd_time,3), "s ] has shape",S_hd.shape,"and rank",rank_S_hd,":\n",S_hd, "\n")
		print(name," : Low-Dimension Covariance Matrix [",np.round(S_ld_time,3), "s ] has shape",S_ld.shape,"and rank",rank_S_ld,":\n",S_ld, "\n")

	return S_hd, S_ld
	
def calculate_eigenvectors_eigenvalues(covariance_matrix_hd, covariance_matrix_ld,A_train,mode):
	start_time = time.time()
	eigenvalues_hd, eigenvectors_hd = eigh(covariance_matrix_hd) 
	eigenvalues_eigenvectors_hd_time = time.time() - start_time
	idx_hd = eigenvalues_hd.argsort()[::-1]
	eigenvalues_hd = eigenvalues_hd[idx_hd]
	eigenvectors_hd = eigenvectors_hd[:,idx_hd]
	start_time = time.time()
	eigenvalues_ld, eigenvectors_ld = eigh(covariance_matrix_ld) 
	eigenvalues_eigenvectors_ld_time = time.time() - start_time
	idx_ld = eigenvalues_ld.argsort()[::-1]
	eigenvalues_ld = eigenvalues_ld[idx_ld]
	eigenvectors_ld = eigenvectors_ld[:,idx_ld]
	V = eigenvectors_ld										# V = matrix of low-dimension eigenvectors
	_U = np.dot(A_train,V)								# reconstructed U from low-dimension eigenvectors
	if mode == 'print':
		print("High-Dimension Eigenvector = U has shape",eigenvectors_hd.shape,":\n",eigenvectors_hd, "\n")
		print("Low-Dimension Eigenvector = V has shape",eigenvectors_ld.shape,":\n",eigenvectors_ld, "\n")
		
		print("High-Dimension Eigenvalues has shape",eigenvalues_hd.shape,":\n",eigenvalues_hd, "\n")
		print("Low-Dimension Eigenvalues has shape",eigenvalues_ld.shape,":\n",eigenvalues_ld, "\n")
		
	calculate_non_zero_eigenvalues(eigenvalues_hd,"HD Eigenvalues")
	calculate_non_zero_eigenvalues(eigenvalues_ld,"LD Eigenvalues")

	return eigenvalues_hd, eigenvectors_hd, eigenvalues_ld, eigenvectors_ld, V, _U, eigenvalues_eigenvectors_hd_time, eigenvalues_eigenvectors_ld_time

def calculate_combined_eigenvectors_eigenvalues(covariance_matrix,mode):
	start_time = time.time()
	eigenvalues, eigenvectors = eigh(covariance_matrix) 
	eigenvalues_eigenvectors_time = time.time() - start_time
	idx = eigenvalues.argsort()[::-1]
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]

	#V = eigenvectors_ld										# V = matrix of low-dimension eigenvectors
	#_U = np.dot(A_train,V)								# reconstructed U from low-dimension eigenvectors
	
	if mode == 'print':
		print("Combined Eigenvector has shape",eigenvectors.shape,":\n",eigenvectors, "\n")
		print("Combined Eigenvalues has shape",eigenvalues.shape,":\n",eigenvalues, "\n")

	return eigenvalues, eigenvectors
	
def plot_top_M_eigenfaces(eigenvec,M):
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
	overall_title = str("Top "+str(M)+" Eigenfaces for High-Dim PCA")
	plt.suptitle(overall_title)
	plt.show()
	plt.close()

def plot_eigenvalues(eigenvalues_hd, eigenvalues_ld, mode):		# function for plotting high-dimension and low-dimension eigenvalues
	df_hd = DataFrame(eigenvalues_hd, columns = ['High-Dimension Eigenvalues'])	# converting high-dim array into a dataframe
	df_ld = DataFrame(eigenvalues_ld, columns = ['Low-Dimension Eigenvalues'])	# converting the low-dim array into a dataframe
	hd_eigenvalues = df_hd['High-Dimension Eigenvalues']
	ld_eigenvalues = df_ld['Low-Dimension Eigenvalues']
	
	if mode == 'plot':
		plt.figure(figsize=(8,8))
		plt.scatter(df_hd.index.values, hd_eigenvalues, color = 'red', s=50, alpha=1, marker = 'o')	# creating a scatter plot for the high-dim eigenvalues
		plt.scatter(df_ld.index.values, ld_eigenvalues, color = 'blue', s=50, alpha=1, marker = 'x')	# creating a scatter plot for the low-dim eigenvalues
		plt.plot(df_hd.index.values, hd_eigenvalues, color = 'red')									# creating line graph for high-dim eigenvalues
		plt.plot(df_ld.index.values, ld_eigenvalues, color = 'blue')									# creating line graph for lower-dim eigenvalues

		plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)							# parameters for plot grid
		plt.xticks(np.arange(min(df_hd.index.values),max(df_hd.index.values)+250, 250))							# adjusting the intervals to 250
		#plt.yticks(np.arange(min(hd_eigenvalues),max(hd_eigenvalues)+25000, 50000))			# adjusting the intervals to 50,000
		plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))						# setting y-axis to scientific scale
		plt.title('Comparing High-Dimension and Low-Dimension PCA Eigenvalues')

		plt.xlabel('Eigenvalue Index (M)')
		plt.ylabel('Eigenvalue')			
		plt.legend(loc = 'upper right')														# creating legend and placing in at the top right
		plt.show()	
		
	return df_hd, df_ld
	
def reconstruct_image_LD_PCA(X_train,Y_train,A,_U,sample_list,M_list,average_training_face,main_title,mode):
	if mode == 'time':
		sample_list = [0]
		M_list = [10,20,30,40,50,60,70,80,90,100,200,300,400,415,416]
	if mode == 'error':
		sample_list = [0]
		M_list = [10,20,30,40,50,60,70,80,90,100,200,300,400,415,416]
	index = 1
	font_size = 10
	count = 1
	cols = len(M_list)
	rows = len(sample_list)
	plt.figure(figsize=(20,10))
	for sample in sample_list:
		M_time_list = []
		reconstruction_time_list = []
		reconstruction_error_list = []
		original_face = X_train[sample]		
		original_face = np.array(original_face)[np.newaxis]
		plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(original_face,(46,56)).T, cmap = 'gist_gray')
		title = str("F"+str(Y_train[sample])+"-Original")
		plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
		index=index+1
		for M in M_list:
			start_time = time.time()
			__U = normalize(_U,axis=0,norm='l2')
			#__U = np.multiply(__U,-1)				# same results even if you do not multiply with -1
			__U = __U[:,0:M]
			W = np.dot(A.T,__U)	
			reconstructed_image = average_training_face
			for i in range(0,M):
				print(main_title, ": F",Y_train[sample], "[", count, "/", rows, "] with M =", M, "... ( M = ", i+1, "/", M, ")    ", end="\r")
				au = W[sample,i]*__U[:,i]
				reconstructed_image = reconstructed_image + au
			reconstruction_time = time.time() - start_time	
			M_time_list.append(M)
			#reconstructed_image = np.array([round(x) for x in reconstructed_image.squeeze()])
			reconstructed_image = np.array([int(x) for x in reconstructed_image.squeeze()])
			reconstruction_error = np.subtract(original_face,reconstructed_image)
			reconstruction_error = norm(reconstruction_error)/norm(original_face)*100
			reconstruction_error_list.append(reconstruction_error)
			reconstruction_time_list.append(reconstruction_time)
			title = str("F"+str(Y_train[sample])+" [M=" + str(M) + "]")
			plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(reconstructed_image,(46,56)).T, cmap = 'gist_gray')
			plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
			index = index+1
		count = count+1
	if mode == 'time':
		time_df_ld = DataFrame(M_time_list, columns = ['M'])	# converting high-dim array into a dataframe
		time_df_ld['Reconstruction Time(LD)'] = reconstruction_time_list
		reconstruction_df_ld = None
	elif mode == 'error':
		reconstruction_df_ld = DataFrame(M_time_list, columns = ['M'])	
		reconstruction_df_ld['Reconstruction Error(LD)'] = reconstruction_error_list
		time_df_ld = None
	else:
		time_df_ld = None
		reconstruction_df_ld = None
	plt.suptitle(main_title)
	if mode == 'save':
		plt.savefig(main_title)
	if mode == 'show':
		plt.show()
	plt.close()
	
	return time_df_ld, reconstruction_df_ld
	
def reconstruct_image_HD_PCA(X_train,Y_train,A,eigenvectors_hd,sample_list,M_list,average_training_face,main_title,mode):
	if mode == 'time':
		sample_list = [0]
		M_list = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,208]
	if mode == 'error':
		sample_list = [0]
		M_list = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,208]
	index = 1
	font_size = 8
	count = 1
	cols = len(M_list)
	rows = len(sample_list)
	plt.figure(figsize=(20,10))
	for sample in sample_list:
		M_time_list = []
		reconstruction_time_list = []
		reconstruction_error_list = []
		original_face = X_train[sample]		
		original_face = np.array(original_face)[np.newaxis]
		plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(original_face,(46,56)).T, cmap = 'gist_gray')
		title = str("F"+str(Y_train[sample])+"-Original")
		plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
		index=index+1
		for M in M_list:
			start_time = time.time()
			U = eigenvectors_hd[:,0:M]
			print("U =",U.shape)
			print("A.T =",A.T.shape)
			W = np.dot(A.T,U)
			reconstructed_image = average_training_face
			for i in range(0,M):
				print(main_title, ": F",Y_train[sample], "[", count, "/", rows, "] with M =", M, "... ( M = ", i+1, "/", M, ")    ", end="\r")
				au = W[sample,i]*U[:,i]
				reconstructed_image = reconstructed_image + au
			reconstruction_time = time.time() - start_time	
			M_time_list.append(M)
			#reconstructed_image = np.array([round(x) for x in reconstructed_image.squeeze()])
			reconstructed_image = np.array([int(x) for x in reconstructed_image.squeeze()])
			reconstruction_error = np.subtract(original_face,reconstructed_image)
			reconstruction_error = norm(reconstruction_error)/norm(original_face)*100
			reconstruction_error_list.append(reconstruction_error)
			reconstruction_time_list.append(reconstruction_time)
			title = str("F"+str(Y_train[sample])+" [M=" + str(M) + "]")
			plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(reconstructed_image,(46,56)).T, cmap = 'gist_gray')
			plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
			index = index+1
		count = count+1

	if mode == 'time':
		time_df_hd = DataFrame(M_time_list, columns = ['M'])	
		time_df_hd['Reconstruction Time(HD)'] = reconstruction_time_list
		reconstruction_df_hd = None
	elif mode == 'error':
		reconstruction_df_hd = DataFrame(M_time_list, columns = ['M'])	
		reconstruction_df_hd['Reconstruction Error(HD)'] = reconstruction_error_list
		time_df_hd = None
	else:
		time_df_hd = None
		reconstruction_df_hd = None
	plt.suptitle(main_title)
	if mode == 'save':
		plt.savefig(main_title)
	if mode == 'show':
		plt.show()
	plt.close()
	
	return time_df_hd, reconstruction_df_hd

def plot_time(df_time_hd, df_time_ld,mode):									# function for plotting high-dimension and low-dimension eigenvalues
	time_hd = df_time_hd['Reconstruction Time(HD)']
	M_hd = df_time_hd['M']
	time_ld = df_time_ld['Reconstruction Time(LD)']
	M_ld = df_time_ld['M']

	plt.figure(figsize=(8,8))
	plt.plot(M_hd, time_hd, color = 'red')									
	plt.plot(M_ld, time_ld, color = 'blue')									
	plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)				# parameters for plot grid
	plt.xticks(np.arange(0,max(M_hd)+50, 250))								# adjusting the intervals to 250
	plt.title('Comparing High-Dimension and Low-Dimension PCA Reconstruction Time').set_position([0.5,1.05])
	plt.xlabel('M')
	plt.ylabel('Time Taken(s)')			
	plt.legend(loc = 'upper left')											# creating legend and placing in at the top right
	if mode == 'save':
		plt.savefig('Comparing High-Dimension and Low-Dimension PCA Reconstruction Time')
	if mode == 'show':
		plt.show()
	plt.close()
	plt.figure(figsize=(8,8))
	plt.plot(M_hd, time_hd, color = 'red')									
	plt.plot(M_ld, time_ld, color = 'blue')									
	plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)				# parameters for plot grid
	plt.xticks(np.arange(0,max(M_hd)+50, 250))								# adjusting the intervals to 250
	plt.title('Comparing High-Dimension and Low-Dimension PCA Reconstruction Time').set_position([0.5,1.05])
	plt.xlabel('M')
	plt.ylabel('Time Taken(s)')			
	plt.legend(loc = 'upper left')											# creating legend and placing in at the top right
	plt.xticks(np.arange(0,max(M_hd), 50))
	plt.xlim([0,416])	
	plt.ylim([0,0.05])	

	if mode == 'save':
		plt.savefig('Comparing High-Dimension and Low-Dimension PCA Reconstruction Time (Zoomed to M=416)')
	if mode == 'show':
		plt.show()
	plt.close()

def plot_reconstruction_error(df_time_hd, df_time_ld,mode):									# function for plotting high-dimension and low-dimension eigenvalues
	time_hd = df_time_hd['Reconstruction Error(HD)']
	M_hd = df_time_hd['M']
	time_ld = df_time_ld['Reconstruction Error(LD)']
	M_ld = df_time_ld['M']

	plt.figure(figsize=(8,8))
	plt.plot(M_hd, time_hd, color = 'red')									
	plt.plot(M_ld, time_ld, color = 'blue')									
	plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)				# parameters for plot grid
	plt.xticks(np.arange(0,max(M_hd)+50, 250))								# adjusting the intervals to 250
	plt.yticks(np.arange(0,max(time_hd)+2, 1))
	plt.title('Comparing High-Dimension and Low-Dimension Reconstruction Error').set_position([0.5,1.05])
	plt.xlabel('M')
	plt.ylabel('Reconstruction Error(%)')			
	plt.legend(loc = 'upper right')											# creating legend and placing in at the top right
	if mode == 'save':
		plt.savefig('Comparing High-Dimension and Low-Dimension Reconstruction Error')
	if mode == 'show':
		plt.show()
	plt.close()
	plt.figure(figsize=(8,8))
	plt.plot(M_hd, time_hd, color = 'red')									
	plt.plot(M_ld, time_ld, color = 'blue')									
	plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)				# parameters for plot grid
	plt.xticks(np.arange(0,max(M_hd)+50, 250))								# adjusting the intervals to 250
	plt.title('Comparing High-Dimension and Low-Dimension PCA Reconstruction Time').set_position([0.5,1.05])
	plt.xlabel('M')
	plt.ylabel('Reconstruction Error(%)')			
	plt.legend(loc = 'upper right')											# creating legend and placing in at the top right
	plt.xticks(np.arange(0,max(M_hd), 50))
	plt.yticks(np.arange(0,max(time_hd)+2, 1))
	plt.xlim([0,416])		

	if mode == 'save':
		plt.savefig('Comparing High-Dimension and Low-Dimension PCA Reconstruction Time (Zoomed to M=416)')
	if mode == 'show':
		plt.show()
	plt.close()

def LD_NN_classifier(M_list_NN,A_train,A_test,X_test,Y_train,Y_test,V,average_training_face,mode,save):
	test_size = len(Y_test)
	train_size = len(Y_train)
	if mode == 'plot time':
		M_list = [10,20,30,40,50,60,70,80,90,100,200,300,400,415,416]
	if mode == 'success':
		M_list_NN = [30,60,90,416]
	if mode == 'fail':
		M_list_NN = [30,60,90,416]
	success_rate_list = []
	success_rate_M_list = []
	M_time_list = []
	recognition_time_list = []
	mem_psutil = []
	previous = 0
	success_rate_list.append(0)
	success_rate_M_list.append(0)
	for M in M_list_NN:
	#for M in range(0,416,10):
		start_time = time.time()
		success = 0
		_U = np.dot(A_train,V)	
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
				font_size=10
				if mode == 'success':
					index1=1
					index2=0
					index3=0
					plt.figure(figsize=(20,10))
					for M in M_list_NN:
						_U = np.dot(A_train,V)	
						__U = normalize(_U,axis=0,norm='l2')
						__U = np.multiply(__U,-1)
						__U = __U[:,0:M]
						W_train = np.dot(A_train.T,__U)		
						W_test = np.dot(A_test.T,__U)
						font_size = 10
						reconstructed_train_image = average_training_face
						reconstructed_test_image = average_training_face
						for i in range(0,M):
							au_train = W_train[pos,i]*__U[:,i]
							au_test = W_test[ntest,i]*__U[:,i]
							reconstructed_train_image = reconstructed_train_image + au_train
							reconstructed_test_image = reconstructed_test_image + au_test
						title1 = str("Reconstructed Train: F"+str(Y_train[pos])+", M="+str(M))
						title2 = str("Reconstructed Test: F"+str(Y_test[ntest])+", M="+str(M))
						title3 = str("Original Test")
						index2=index1+4
						index3=index1+8
						plt.subplot(3,4,index1), plt.imshow(np.reshape(reconstructed_train_image,(46,56)).T, cmap = 'gist_gray')
						plt.title(title1, fontsize=font_size).set_position([0.5,1.0]), plt.xticks([]), plt.yticks([])
						plt.subplot(3,4,index2), plt.imshow(np.reshape(reconstructed_test_image,(46,56)).T, cmap = 'gist_gray')
						plt.title(title2, fontsize=font_size).set_position([0.5,1.0]), plt.xticks([]), plt.yticks([])
						plt.subplot(3,4,index3), plt.imshow(np.reshape(np.array(X_test[ntest])[np.newaxis],(46,56)).T, cmap = 'gist_gray')
						plt.title(title3, fontsize=font_size).set_position([0.5,1.0]), plt.xticks([]), plt.yticks([])
						plt.suptitle("Success Example")
						index1 = index1+1
					if save == 'yes':
						file_title = str("Success Example")
						plt.savefig(file_title)
						plt.close()
					else:
						plt.show()
						plt.close()
					mode = 'none'
					
			if Y_test[ntest] != label:
				font_size=10
				if mode == 'fail':
					index1=1
					index2=0
					index3=0
					plt.figure(figsize=(20,10))
					for M in M_list_NN:
						_U = np.dot(A_train,V)	
						__U = normalize(_U,axis=0,norm='l2')
						__U = np.multiply(__U,-1)
						__U = __U[:,0:M]
						W_train = np.dot(A_train.T,__U)		
						W_test = np.dot(A_test.T,__U)
						font_size = 10
						reconstructed_train_image = average_training_face
						reconstructed_test_image = average_training_face
						for i in range(0,M):
							au_train = W_train[pos,i]*__U[:,i]
							au_test = W_test[ntest,i]*__U[:,i]
							reconstructed_train_image = reconstructed_train_image + au_train
							reconstructed_test_image = reconstructed_test_image + au_test
						title1 = str("Reconstructed Train: F"+str(Y_train[pos])+", M="+str(M))
						title2 = str("Reconstructed Test: F"+str(Y_test[ntest])+", M="+str(M))
						title3 = str("Original Test")
						index2=index1+4
						index3=index1+8
						plt.subplot(3,4,index1), plt.imshow(np.reshape(reconstructed_train_image,(46,56)).T, cmap = 'gist_gray')
						plt.title(title1, fontsize=font_size).set_position([0.5,1.0]), plt.xticks([]), plt.yticks([])
						plt.subplot(3,4,index2), plt.imshow(np.reshape(reconstructed_test_image,(46,56)).T, cmap = 'gist_gray')
						plt.title(title2, fontsize=font_size).set_position([0.5,1.0]), plt.xticks([]), plt.yticks([])
						plt.subplot(3,4,index3), plt.imshow(np.reshape(np.array(X_test[ntest])[np.newaxis],(46,56)).T, cmap = 'gist_gray')
						plt.title(title3, fontsize=font_size).set_position([0.5,1.0]), plt.xticks([]), plt.yticks([])
						plt.suptitle("Fail Example")
						index1 = index1+1
					if save == 'yes':
						file_title = str("Fail Example")
						plt.savefig(file_title)
						plt.close()
					else:
						plt.show()
						plt.close()
					mode = 'none'

			recognition_time = time.time() - start_time	
		success_rate = success/test_size*100
		increase = success_rate - previous
		previous = success_rate
		recognition_time_list.append(recognition_time)
		M_time_list.append(M)
		success_rate_list.append(success_rate)
		success_rate_M_list.append(M)
		recognition_time_list.append(recognition_time)
		mem_psutil.append(memory_usage_psutil())
		if mode == 'print':
			print("For M =",M,"success rate is",success_rate,"% ... increase = ",increase)
	if mode == 'plot mem':
		df = DataFrame(M_time_list, columns = ['M'])	# converting high-dim array into a dataframe
		df['psutil'] = mem_psutil
		x = df['M']
		y = df['psutil']
		plt.figure(figsize=(8,8))
		plt.plot(x,y,color = 'black')
		plt.xticks(np.arange(0,max(x)+50,250))								# adjusting the intervals to 250	
		plt.title('Relationship between M and Memory Usage').set_position([0.5,1.05])
		plt.xlabel('M')
		plt.ylabel('Memoru Usage in MB')			
		plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)		
		plt.legend(loc = 'upper left')											# creating legend and placing in at the top right
		if save == 'yes':
			plt.savefig('Comparing High-Dimension and Low-Dimension PCA Reconstruction Time')
		else:
			plt.show()	
	if mode == 'plot time':
		df = DataFrame(M_time_list, columns = ['M'])	# converting high-dim array into a dataframe
		df['Recognition Time'] = recognition_time_list
		x = df['M']
		y = df['Recognition Time']
		plt.figure(figsize=(8,8))
		plt.plot(x,y,color = 'black')																	
		plt.xticks(np.arange(0,max(x)+50, 250))								# adjusting the intervals to 250
		plt.title('Relationship between M and Recognition Time').set_position([0.5,1.05])
		plt.xlabel('M')
		plt.ylabel('Time Taken(s)')			
		plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)		
		plt.legend(loc = 'upper left')											# creating legend and placing in at the top right
		if save == 'yes':
			plt.savefig('Comparing High-Dimension and Low-Dimension PCA Reconstruction Time')
		else:
			plt.show()		
	if mode == 'plot rate':
		df = DataFrame(success_rate_M_list, columns = ['M'])	# converting high-dim array into a dataframe
		df['Success Rate'] = success_rate_list
		x = df['M']
		y = df['Success Rate']
		
		fig,ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(x,y,color = 'red')
		ax2.plot(df_hd.index.values,df_hd['High-Dimension Eigenvalues'], color = 'green')
		ax1.set_xlabel('M')
		ax1.set_ylim([0,1.01*max(y)])
		ax2.set_ylim([0,1.01*max(df_hd['High-Dimension Eigenvalues'])])
		ax1.set_xlim([-10,100])
		ax1.grid()
		ax1.set_ylabel('Success Rate(%)',color='red')
		ax2.set_ylabel('Eigenvalues',color='green')	
		plt.suptitle('Comparing Top Eigenvalues and Success Rate against M')
		if save == 'yes':
			plt.savefig('Comparing Top Eigenvalues and Success Rate against M')
		else:
			plt.show()
		plt.close()
		
		plt.figure(figsize=(8,8))
		plt.scatter(x, y, color = 'red', s=50, alpha=1, marker = 'x')	# creating a scatter plot for the high-dim eigenvalues
		plt.plot(x,y, color = 'black')									# creating line graph for high-dim eigenvalues
		plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)							# parameters for plot grid
		plt.xticks(np.arange(0,max(x)+250, 250))							# adjusting the intervals to 250
		plt.yticks(np.arange(0,100, 10))			# adjusting the intervals to 50,000
		plt.title('Relationship between M and Success Rate')
		plt.xlabel('M)')
		plt.ylabel('Success Rate (%)')			
		plt.legend(loc = 'upper right')														# creating legend and placing in at the top right
		if save == 'yes':
			plt.savefig('Relationship between M and Success Rate')
		else:
			plt.show()
		plt.close()
		
######################################### MAIN STARTS HERE ###########################################################	
def main():	
	#### LOAD FACE DATA
	mat_content = loadmat('face(1).mat')			# unpacks the .mat file
	print("\n")
	#print("Showing contents of the .mat file:")		# shows the contents of the .mat file
	#print(mat_content)
	face_data = mat_content['X']
	face_labels = mat_content['l']	

	#### PARTITION DATA INTO TRAIN AND TEST SET
	x_train, y_train, x_test, y_test, train_size, test_size = partition_data(face_data,face_labels,split=0.2,mode='none')
	X_train = x_train.squeeze()				# This is the matrix that contains all of the face training data
	X_test = x_test.squeeze()		
	Y_train = y_train						# This is the matrix that contains all of the labels for the training data
	Y_test = y_test			

	#### PARTITION TRAIN DATA INTO 4 SUBSETS
	x_subsets, y_subsets = divide_training_data_into_4_subsets(x_train,y_train,train_size,mode='none')
	x_subset1 = x_subsets[0]
	x_subset2 = x_subsets[1]
	x_subset3 = x_subsets[2]
	x_subset4 = x_subsets[3]
	y_subset1 = y_subsets[0]
	y_subset2 = y_subsets[1]
	y_subset3 = y_subsets[2]
	y_subset4 = y_subsets[3]

	#### CALCULATE AVERAGE TRAINING FACES
	batch_mean_face = calculate_mean_image(x_train.squeeze(),name='Batch',mode='none')	# mode = 'print'/'none'
	subset1_mean_face = calculate_mean_image(x_subset1,name='Subset1',mode='none')		# u1
	subset2_mean_face = calculate_mean_image(x_subset2,name='Subset2',mode='none')		# u2
	subset3_mean_face = calculate_mean_image(x_subset3,name='Subset3',mode='none')		# u3
	subset4_mean_face = calculate_mean_image(x_subset4,name='Subset4',mode='none')		# u4
	subset_mean_list = [subset1_mean_face,subset2_mean_face,subset3_mean_face,subset4_mean_face]
	
	#### CALCULATE DESIRED SUBSET COMBINED MEAN
	subset_list = [x_subset1,x_subset2]										# insert subsets to find mean, i.e. putting subset1 and subset2 will calculate the combined mean of subset1 and subset2
	combined_subset_mean = calculate_combined_mean(subset_list,subset_mean_list,mode='none')
	
	#### MERGE AND CALCULATE EQUIVALENT BATCH MEAN
	x_combined, batch_mean = calculate_equivalent_batch_mean(subset_list,mode='none')
	A_subset1 = np.subtract(x_subset1,subset1_mean_face).squeeze().T
	A_subset2 = np.subtract(x_subset2,subset2_mean_face).squeeze().T
	A_combined = np.subtract(x_combined,combined_subset_mean).squeeze()
	A_combined = A_combined.T	
	#print(A_combined)
	#print(A_combined.shape)
	
	#### CALCULATE COMBINED COVARIANCE MATRIX: SUBSET_1 + SUBSET_2
	S_combined = calculate_combined_covariance(x_subset1,x_subset2,subset1_mean_face,subset2_mean_face,mode='none')
	S3 = S_combined
	
	#### CALCULATE BATCH COVARIANCE MATRIX AND EIGENS
	A_batch,A_test,S_batch_hd,S_batch_ld, S_batch_hd_time, S_batch_ld_time = calculate_covariance_matrix(x_combined,x_test,batch_mean,mode='none')
	eigenvalues_batch_hd, eigenvectors_batch_hd, eigenvalues_batch_ld, eigenvectors_batch_ld, V, _U, eigenvalues_eigenvectors_hd_time, eigenvalues_eigenvectors_ld_time = calculate_eigenvectors_eigenvalues(S_batch_hd, S_batch_ld,A_batch,mode='none')

	sys.exit()
	#### CALCULATE P1, P2, u1-u2
	S1_hd, S1_ld = calculate_subset_covariance_matrix(x_subset1,subset1_mean_face,name='Subset1',mode='none') # mode = 'print'/'none'
	S2_hd, S2_ld = calculate_subset_covariance_matrix(x_subset2,subset2_mean_face,name='Subset2',mode='none') # mode = 'print'/'none'
	eigenvaluesP1, P1 = calculate_combined_eigenvectors_eigenvalues(S1_ld,mode='none')
	eigenvaluesP2, P2 = calculate_combined_eigenvectors_eigenvalues(S2_ld,mode='none')
	u1_u2 = subset1_mean_face - subset2_mean_face
	
	print("P1 =",P1.shape)
	print("P2 =",P2.shape,'\n')
	print("A_subset1 =",A_subset1.shape)
	print("A_subset2 =",A_subset2.shape)
	print("A_combined =",A_combined.shape,'\n')
	print("u1-u2 =",u1_u2.T.shape,'\n')
	
	U1 = np.dot(A_subset1,P1)
	U2 = np.dot(A_subset2,P2)
	print("U1 =",U1.shape)
	print("U2 =",U2.shape,'\n')
	
	P1=U1
	P2=U2
	
	#horizontal_stack_matrix = np.hstack((U1,U2,u1_u2.T))
	#print("horizontal stack =",horizontal_stack_matrix.shape)
	#sys.exit()
	
	#### COMBINE SUBSETS
	x_combined = np.vstack((x_subset1,x_subset2))
	y_combined = y_subset1+y_subset2
	#print("x_combined =",x_combined.shape,'\n',x_combined,'\n')
	#print("y_combined =",len(y_combined),'\n',y_combined,'\n')

	#### STACK P1,P2,u1-u2
	horizontal_stack_matrix = np.hstack((P1,P2,u1_u2.T))
	#vertical_stack_matrix = np.vstack((P1.T,P2.T,u1_u2))
	print("horizontal stack =",horizontal_stack_matrix.shape)
	#print("vertical stack =",vertical_stack_matrix.shape,'\n')
	#print("S3 =",S_combined.shape)
	
	#### CALCULATE PHI
	#PHI_H = orth(horizontal_stack_matrix)
	Q,R = np.linalg.qr(horizontal_stack_matrix)
	#PHI_V = orth(vertical_stack_matrix)
	#print("PHI_H =",PHI_H.shape)
	print("Q =",Q.shape)
	#print("PHI_V =",PHI_V.shape,"\n")
	
	print("S3 =",S3.shape,"\n")
	
	#### CALCULATE (PHI.T)(S3)(PHI)
	PHI = Q
	S3_PHI = np.dot(S3,PHI)
	PHIT_S3_PHI = np.dot(PHI.T,S3_PHI)
	print("(PHI.T)(S3)(PHI) =",PHIT_S3_PHI.shape,"\n",PHIT_S3_PHI,"\n")
	
	#### EIGEN DECOMPOSITION ON (PHI.T)(S3)(PHI)
	eigenvalues_S3, eigenvectors_S3 = calculate_combined_eigenvectors_eigenvalues(PHIT_S3_PHI,mode='none')
	LAMBDA3 = eigenvalues_S3
	R3 = eigenvectors_S3
	
	idx = LAMBDA3.argsort()[::-1]
	LAMBDA3 = LAMBDA3[idx]
	R3 = R3[:,idx]
	
	P3 = np.dot(PHI,R3)
	
	print("LAMBDA3 =",LAMBDA3.shape)
	print("R3 =",R3.shape)
	print("PHI =",PHI.shape)
	print("P3 =",P3.shape,"\n")
	
	sys.exit()
	#### RECONSTRUCTION
	#### PCA RECONSTRUCTION: HIGH- AND LOW-DIMENSION
	#M_list_ld = [10,20,30,40,50,60,70,80,90,100,200,300,400,416]						# list that contains values of M to try for LD PCA Reconstruction
	#M_list_hd = [10,20,30,40,50,60,70,80,90,100,200,300,400,416,1000,1500,2000,2576]	# list that contains values of M to try for HD PCA Reconstruction
	M_list_hd = [10,20,30,40,50,60,70,80,90,100,200,208]
	sample_list = [0,9,17,25,33,41]														# list of face images to try for PCA Reconstruction
	time_df_inc, reconstruction_df_inc = reconstruct_image_HD_PCA(x_combined,y_combined,A_combined,P3,sample_list,M_list_hd,combined_subset_mean,"Incremental PCA Reconstruction", mode='error')		#mode=save/show/time/error
	time_df_batch, reconstruction_df_batch = reconstruct_image_LD_PCA(X_train,Y_train,A_train,eigenvectors_ld,sample_list,M_list_ld,average_training_face,"Low-Dim PCA Reconstruction", mode='show')		#mode=save/show/time/error
	#plot_time(time_df_hd, time_df_ld,mode='show')										# mode='save'/'show', reconstruct_image_HD/LD_PCA(mode='time')
	plot_reconstruction_error(reconstruction_df_inc, reconstruction_df_batch,mode='show')	# mode='save'/'show', reconstruct_image_HD/LD_PCA(mode='error')

	
	
	
	
	sys.exit()
	
	return 0
	
main()
