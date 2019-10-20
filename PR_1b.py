### Data provided: face(1).mat
### Partition the provided face data face(1).mat into training and testing data in a way you choose
### Explain briefly the way you partitioned

### Apply PCA to your training data
### i.e. Compute the eigenvectors and eigenvalues of the Covariance Matrix S = (1/N)AA^T directly

### Show and discuss:
### 1. eigenvectors and eigenvalues 
### 2. mean image
### 3. eigenvectors with non-zero values
### 4. how many eigenvectors are to be used for face recognition

### Give physical meanings behind your answers

# Importing depdendencies
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import cv2
import sys
import time
import os

from scipy.io import loadmat
from sklearn.decomposition import PCA
										# comment out this code when running on jupyter
dir = os.getcwd()								# gets the working directory of the file this python script is in
os.chdir (dir)									# changes the working directory to the file this python script is in
										# also change plt.savefig to plt.show
										
def print_image(face_image):							# function for plotting an image
	face_image = np.reshape(face_image, (46,56))
	face_image = face_image.T
	plt.imshow(face_image, cmap='gist_gray')
	plt.show()

def save_image(face_image,title):						# function for saving an image
	face_image = np.reshape(face_image, (46,56))
	face_image = face_image.T
	plt.imshow(face_image, cmap='gist_gray')
	plt.title(title)
	plt.savefig(title)

def plot_eigenvalues(df_hd, df_ld):						# function for plotting high-dimension and low-dimension eigenvalues
	hd_index = df_hd['HD Index']
	hd_eigenvalues = df_hd['High-Dimension Eigenvalues']
	ld_index = df_ld['LD Index']
	ld_eigenvalues = df_ld['Low-Dimension Eigenvalues']

	plt.figure(figsize=(8,8))
	plt.scatter(hd_index, hd_eigenvalues, color = 'red', s=50, alpha=1, marker = 'o')	# creating a scatter plot for the high-dim eigenvalues
	plt.scatter(ld_index, ld_eigenvalues, color = 'blue', s=50, alpha=1, marker = 'x')	# creating a scatter plot for the low-dim eigenvalues
	plt.plot(hd_index, hd_eigenvalues, color = 'red')					# creating line graph for high-dim eigenvalues
	plt.plot(ld_index, ld_eigenvalues, color = 'blue')					# creating line graph for lower-dim eigenvalues

	plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)				# parameters for plot grid
	plt.xticks(np.arange(min(hd_index),max(hd_index)+250, 250))				# adjusting the intervals to 250
	plt.yticks(np.arange(min(hd_eigenvalues),max(hd_eigenvalues)+25000, 50000))		# adjusting the intervals to 50,000
	plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))				# setting y-axis to scientific scale
	plt.title('Comparing High-Dimension and Low-Dimension PCA Eigenvalues')

	plt.xlabel('Eigenvalue Index (M)')
	plt.ylabel('Eigenvalue')			
	plt.legend(loc = 'upper right')														# creating legend and placing in at the top right
	plt.show()	

mat_content = loadmat('face(1).mat')			# unpacks the .mat file

print("\n")
print("Showing contents of the .mat file:")		# shows the contents of the .mat file
print(mat_content)

# Array 'X' contains the face data	
# Each column represents one face image
# Each row represents a pixel value for a particular coordinate of the image
face_data = mat_content['X']
face_labels = mat_content['l']	

print("\n")
print("Showing Face Data = mat_content['X']:")					
print(face_data)

print("\nSize of Face Data: ", face_data.shape)
print("Number of rows (Pixel Values):", face_data.shape[0])
print("Number of columns (Face Images):", face_data.shape[1], "\n")

number_of_faces = face_data.shape[1]
number_of_pixels = face_data.shape[0]

face_labels_array = []
face_pixels_array = []

# Create an array that only contains the column pixels of the image
for i in range (0,number_of_faces):
	face_labels_array.append(face_labels[0,i])					# Array that contains the labels
	face_pixels_array.append(face_data[:,i])					# Array that contrains the face_data
	#print(face_labels_array[i], " = ", face_pixels_array[i])

#### PARTITIONING FACE DATA INTO TRAINING AND TESTING SET
split = 0.2		# split train:test = 0.8:0.2
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(face_pixels_array, face_labels_array, test_size=split, random_state=42)

print("Split = ", split, " ---> Train =", 1-split, ", Test =", split)
print("Size of Train Data (Pixel Vectors): ", 	len(x_train))
print("Size of Train Data (Labels)", 			len(y_train))
print("Size of Test Data (Pixel Vectors): ", 	len(x_test))
print("Size of Test Data (Labels)", 			len(y_test), "\n")

train_size = len(x_train)
test_size = len(x_test)

x_train = np.array(x_train)[np.newaxis]

#### MEAN IMAGE
sum_of_training_faces =[0]
for i in range(0,train_size):
	sum_of_training_faces = sum_of_training_faces + x_train[:,i]

print("Number of train face added: ", i+1)
print("Sum of Training Faces = ", sum_of_training_faces, "\n")

average_training_face = sum_of_training_faces/train_size
print("Mean Face = ", average_training_face, "\n")

#print_image(average_training_face)
save_image(average_training_face,"Average Training Face")

#### COVARIANCE MATRIX
A = []

start_time = time.time()
for i in range(0,train_size):
	phi = x_train[:,i]-average_training_face
	phi = np.array(phi)

	if i == 0:
		A = [phi]
		A = np.array(A)
		A = A.squeeze()
		A = np.array(A)[np.newaxis]
	if i > 0:	
		A = np.append(A, phi, axis=0)
A_time = time.time()-start_time

start_time_hd = time.time()
covariance_matrix_hd = np.dot(A.T,A)/train_size
covariance_matrix_hd_time = time.time() - start_time_hd 

start_time_ld = time.time()
covariance_matrix_ld = np.dot(A,A.T)/train_size
covariance_matrix_ld_time = time.time() - start_time_ld

print("Time taken to obtain High-Dim Covariance Matrix: ", A_time+covariance_matrix_hd_time, "s")
print("Time taken to obtain Low-Dim Covariance Matrix: ", A_time+covariance_matrix_ld_time, "s \n")

print("Covariance Matrix (High-Dimension) has shape: ", covariance_matrix_hd.shape)
print("Covariance Matrix (Low-Dimension) has shape: ", covariance_matrix_ld.shape, "\n")

print("Covariance Matrix (High-Dimension) = ")
print(covariance_matrix_hd, "\n")
print("Covariance Matrix (Low-Dimension) = ")
print(covariance_matrix_ld, "\n")

#### EIGENVALUES AND EIGENVECTORS
start_time=time.time()
eigenvalues_hd, eigenvectors_hd = np.linalg.eig(covariance_matrix_hd)	# obtain eigenvalues and eigenvectors from the high-dimension covariance matrix
eigenvalues_hd_eigenvectors_hd_time = time.time()-start_time
start_time=time.time()
eigenvalues_ld, eigenvectors_ld = np.linalg.eig(covariance_matrix_ld)	# obtain eigenvalues and eigenvectors from the low-dimension covariance matrix
eigenvalues_ld_eigenvectors_ld_time = time.time()-start_time

print("High-Dimension eigenvalues = ")
print(eigenvalues_hd, "\n")
print("Low-Dimension eigenvalues = ")
print(eigenvalues_ld, "\n")

print("High-Dimension Eigenvectors = ")
print(eigenvectors_hd, "\n")
print("Low-Dimension Eigenvectors = ")
print(eigenvectors_ld, "\n")

print("Time taken to obtain High-Dim Eigenvalues and Eigenvectors: ", eigenvalues_hd_eigenvectors_hd_time, "s")
print("Time taken to obtain Low-Dim Eigenvalues and Eigenvectors: ", eigenvalues_ld_eigenvectors_ld_time, "s \n")

#### SORT EIGENVALUES AND EIGENVECTORS FROM HIGHEST TO LOWEST
idx_hd = eigenvalues_hd.argsort()[::-1]
eigenvalues_hd = eigenvalues_hd[idx_hd]
eigenvectors_hd = eigenvectors_hd[:,idx_hd]

idx_ld = eigenvalues_ld.argsort()[::-1]
eigenvalues_ld = eigenvalues_ld[idx_ld]
eigenvectors_ld = eigenvectors_ld[:,idx_ld]

print("Number of High-Dimension eigenvalues: ", eigenvalues_hd.shape[0])
print("Number of High-Dimension eigenvectors: ", eigenvectors_hd.shape, "\n")
print("Number of Low-Dimension eigenvalues: ", eigenvalues_ld.shape[0])
print("Number of Low-Dimension eigenvectors: ", eigenvectors_ld.shape, "\n")

number_of_eigenvalues_hd = eigenvalues_hd.shape[0]
number_of_eigenvalues_ld = eigenvalues_ld.shape[0]

#### COUNT THE NUMBER OF EIGENVECTORS WITH NON-ZERO AND NON-NEGATIVE EIGENVALUES
eigenvalues_hd_array = []
eigenvectors_hd_array = []
eigenvalues_hd_index_array = []
number_of_non_zero_eigenvalues_hd = 0
number_of_non_negative_eigenvalues_hd = 0
for i in range(0, number_of_eigenvalues_hd):
	if eigenvalues_hd[i].real != 0: 
		number_of_non_zero_eigenvalues_hd = number_of_non_zero_eigenvalues_hd + 1
		eigenvalues_hd_array.append(eigenvalues_hd[i].real)
		eigenvectors_hd_array.append(eigenvectors_hd[i])
		eigenvalues_hd_index_array.append(i)
		
	if eigenvalues_hd[i].real > 0:
		number_of_non_negative_eigenvalues_hd = number_of_non_negative_eigenvalues_hd + 1
		
eigenvalues_ld_array = []
eigenvectors_ld_array = []
eigenvalues_ld_index_array = []		
number_of_non_zero_eigenvalues_ld = 0
number_of_non_negative_eigenvalues_ld = 0		
for i in range(0, number_of_eigenvalues_ld):
	if eigenvalues_ld[i].real != 0: 
		number_of_non_zero_eigenvalues_ld = number_of_non_zero_eigenvalues_ld + 1
		eigenvalues_ld_array.append(eigenvalues_ld[i].real)
		eigenvectors_ld_array.append(eigenvectors_ld[i])
		eigenvalues_ld_index_array.append(i)
	
	if eigenvalues_ld[i].real > 0:
		number_of_non_negative_eigenvalues_ld = number_of_non_negative_eigenvalues_ld + 1

print("Number of non-zero High-Dimension Eigenvalues = ", number_of_non_zero_eigenvalues_hd)
print("Number of non-zero Low-Dimension Eigenvalues = ", number_of_non_zero_eigenvalues_ld, "\n")

print("Number of non-negative and non-zero High-Dimension Eigenvalues = ", number_of_non_negative_eigenvalues_hd)
print("Number of non-negative and non-zero Low-Dimension Eigenvalues = ", number_of_non_negative_eigenvalues_ld, "\n")

#### SHOW ALL EIGENVALUES AND EIGENVECTORS
def show_all_eigenvalues_and_eigenvectors():
	
	for i in range(0,number_of_eigenvalues_hd):
		print("High-Dimension Eigenvalue", i+1, " : ", eigenvalues_hd[i])
	print("\n")
	for i in range(0,number_of_eigenvalues_ld):
		print("Low-Dimension Eigenvalue", i+1, " : ", eigenvalues_ld[i].real)
	print("\n")
	for i in range(0,number_of_eigenvalues_hd):
		print("High-Dimension Eigenvectors", i+1, " : ", eigenvectors_hd[i])
	print("\n")
	for i in range(0,number_of_eigenvalues_ld):
		print("Low-Dimension Eigenvectors", i+1, " : ", eigenvectors_ld[i])
	print("\n")	
	
#show_all_eigenvalues_and_eigenvectors()

#### CONVERT EIGENVALUES INTO DATAFRAME TO PLOT 
eigenvalues_hd_array = np.array(eigenvalues_hd_array)								# converting into a numpy array
eigenvectors_hd_array = np.array(eigenvectors_hd_array)
eigenvalues_hd_index_array = np.array(eigenvalues_hd_index_array)

eigenvalues_ld_array = np.array(eigenvalues_ld_array)
eigenvectors_ld_array = np.array(eigenvectors_ld_array)
eigenvalues_ld_index_array = np.array(eigenvalues_ld_index_array)

from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

df_hd = DataFrame(eigenvalues_hd_array, columns = ['High-Dimension Eigenvalues'])	# converting high-dim array into a dataframe
df_hd['HD Index'] = eigenvalues_hd_index_array
df_ld = DataFrame(eigenvalues_ld_array, columns = ['Low-Dimension Eigenvalues'])	# converting the low-dim array into a dataframe
df_ld['LD Index'] = eigenvalues_ld_index_array

combined_df = pd.concat([df_hd, df_ld],axis = 1)									# combining the high-dim and low-dim dataframes into one

#plot_eigenvalues(df_hd, df_ld)

#### PLOTTING MULTI-DIMENSIONAL VECTORS INTO 2D-SPACE
def plot_vectors_into_2D(eigenvectors_hd, eigenvectors_ld):
	pca = PCA(n_components=2)
	reduced_hd = pca.fit_transform(eigenvectors_hd.real)
	reduced_ld = pca.fit_transform(eigenvectors_ld.real)
	t_hd = reduced_hd.transpose()
	t_ld = reduced_ld.transpose()

	plt.figure(figsize=(8,8))
	plt.scatter(t_hd[0], t_hd[1], color = 'red', s=1, alpha=0.8, marker = 'o')
	plt.scatter(t_ld[0], t_ld[1], color = 'blue', s=1, alpha=0.8, marker = 'x')
	plt.show()

def plot_vectors_into_3D(eigenvectors_hd, eigenvectors_ld):
	pca = PCA(n_components=3)
	reduced_hd = pca.fit_transform(eigenvectors_hd.real)
	reduced_ld = pca.fit_transform(eigenvectors_ld.real)
	t_hd = reduced_hd.transpose()
	t_ld = reduced_ld.transpose()

	fig = plt.figure(figsize=(8,8))
	ax = fig.gca(projection='3d')
	ax.scatter(t_hd[0], t_hd[1], t_hd[2], c='red', s=1, alpha = 0.8, marker = 'o')
	ax.scatter(t_ld[0], t_ld[1], t_ld[2], c='blue', s=1, alpha = 0.8, marker = 'x')
	plt.show()

#plot_vectors_into_2D(eigenvectors_hd, eigenvectors_ld)
#plot_vectors_into_3D(eigenvectors_hd, eigenvectors_ld)

#### FACE IMAGE RECONSTRUCTION
X_train = x_train.squeeze()				# This is the matrix that contains all of the face training data
Y_train = y_train						# This is the matrix that contains all of the labels for the training data

training_face0 = X_train[0]
training_face0 = np.array(training_face0)[np.newaxis]

print("A Training Face has shape: ", training_face0.shape)
print("Average Training Face has shape: ", average_training_face.shape, "\n")

#print("Training Labels = ")
#print(Y_train)

M_list_hd = [1,2,3,4,5,6,7,8,9,10,100,416,1000,1497,2576]	# list that contains values of M_hd to try
M_list_ld = [1,2,3,4,5,6,7,8,9,10,100,416]					# list that contains values of M-ld to try
#M_list_hd = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]			# sample list for obtaining a 
#M_list_ld = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
cols_hd = len(M_list_hd)
cols_ld = len(M_list_ld)

sample_list = [0,2,3,4,5,6]
rows = len(sample_list)


def reconstruct_image_PCA(rows,cols,X_train,Y_train,A,eigenvectors,sample_list,M_list,average_training_face,main_title):
	index = 1
	font_size = 10
	count = 1
	plt.figure(figsize=(20,10))
	for sample in sample_list:
		
		face_sample = X_train[sample]							# using training_face[sample] as the training sample
		face_sample = np.array(face_sample)[np.newaxis]
		
		plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(face_sample,(46,56)).T, cmap = 'gist_gray')

		title = str("F"+str(Y_train[sample])+"-Original")
		plt.title(title, fontsize=font_size), plt.xticks([]), plt.yticks([])
		index=index+1
		
		for M in M_list:
			
			M = M
			reconstructed_image = average_training_face
			for i in range(0,M):
				print(main_title, ": F",Y_train[sample], "[", count, "/", rows, "] with M =", M, "... ( M = ", i, "/", M, ")    ", end="\r")
				
				### We have calculated A = [phi_0, phi_1, .... phi_n] where n is the number of training faces, cardinality = nSize
				### A[0] = phi_0, A[1] = phi_1
				phi = np.array(A[sample])[np.newaxis]	

				u = np.array(eigenvectors[i].real)[np.newaxis]
				a = np.dot(phi.T, u)
				au = np.dot(a,u.T)
				reconstructed_image = reconstructed_image + au.T
				
			title = str("F"+str(Y_train[sample])+" [M=" + str(M) + "]")
			plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(reconstructed_image,(46,56)).T, cmap = 'gist_gray')
			plt.title(title, fontsize=font_size), plt.xticks([]), plt.yticks([])
			index = index+1
			
			reconstructed_image = 0
			M = 0
		count = count+1
	plt.suptitle(main_title)
	#plt.show()
	plt.savefig(main_title)
	plt.close()
	
start_time = time.time()
reconstruct_image_PCA(rows,cols_hd,X_train,Y_train,A,eigenvectors_hd,sample_list,M_list_hd,average_training_face,"High-Dim PCA Reconstruction")
high_dim_PCA_reconstruction_time = time.time() - start_time
print("\n")

start_time = time.time()
reconstruct_image_PCA(rows,cols_ld,X_train,Y_train,A,eigenvectors_ld,sample_list,M_list_ld,average_training_face,"Low-Dim PCA Reconstruction")
low_dim_PCA_reconstruction_time = time.time() - start_time
print("\n")

print("High-Dim PCA Reconstruction Elapsed Time = ", np.round(high_dim_PCA_reconstruction_time,3))
print("Low-Dim PCA Reconstruction Elapsed Time = ", np.round(low_dim_PCA_reconstruction_time,3), "\n")

sys.exit()
