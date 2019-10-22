### Plots the top M Eigenfaces

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
from numpy.linalg import norm
										# comment out this code when running on jupyter
dir = os.getcwd()						# gets the working directory of the file this python script is in
os.chdir (dir)							# changes the working directory to the file this python script is in
										# also change plt.savefig to plt.show
										
def print_image(face_image):			# function for plotting an image
	face_image = np.reshape(face_image, (46,56))
	face_image = face_image.T
	plt.imshow(face_image, cmap='gist_gray')
	plt.show()
	plt.close()

def save_image(face_image,title):		# function for saving an image
	face_image = np.reshape(face_image, (46,56))
	face_image = face_image.T
	plt.imshow(face_image, cmap='gist_gray')
	plt.title(title)
	plt.savefig(title)
	plt.close()

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

#### COVARIANCE MATRIX
A = []
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

covariance_matrix_hd = np.dot(A.T,A)/train_size
covariance_matrix_ld = np.dot(A,A.T)/train_size

print("A has shape: ", A.shape, "\n")

print("Covariance Matrix (High-Dimension) has shape: ", covariance_matrix_hd.shape)
print("Covariance Matrix (Low-Dimension) has shape: ", covariance_matrix_ld.shape, "\n")

print("Covariance Matrix (High-Dimension) = ")
print(covariance_matrix_hd, "\n")
print("Covariance Matrix (Low-Dimension) = ")
print(covariance_matrix_ld, "\n")

#### SHOW TOP M EIGENFACES
from scipy.linalg import eigh

max_length_hd = covariance_matrix_hd.shape[0]-1
max_length_ld = covariance_matrix_ld.shape[0]-1

eigenvalues_hd, eigenvectors_hd = eigh(covariance_matrix_hd) 
idx_hd = eigenvalues_hd.argsort()[::-1]
eigenvalues_hd = eigenvalues_hd[idx_hd]
eigenvectors_hd = eigenvectors_hd[:,idx_hd]

eigenvalues_ld, eigenvectors_ld = eigh(covariance_matrix_ld) 
idx_ld = eigenvalues_ld.argsort()[::-1]
eigenvalues_ld = eigenvalues_ld[idx_ld]
eigenvectors_ld = eigenvectors_ld[:,idx_ld]

print("A HD eigenvector has shape ", np.array(eigenvectors_hd[0])[np.newaxis].shape)
print("A LD eigenvector has shape ", np.array(eigenvectors_ld[0])[np.newaxis].shape,"\n")

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

#M = 50
#plot_top_M_eigenfaces(eigenvec_hd,M)

X_train = x_train.squeeze()				# This is the matrix that contains all of the face training data
Y_train = y_train						# This is the matrix that contains all of the labels for the training data
training_face0 = X_train[0]
training_face0 = np.array(training_face0)[np.newaxis]

#M_list_hd = [2576]			
#M_list_ld = [100, 200, 416]
#M_list_hd = [100,200,300,400,416,500,600,700,800,900,1000,1497,2000,2576]		# list that contains values of M_hd to try
M_list_hd = [200,400,416,600,800,1000,1200,1497,1600,1800,2000,2200,2400,2576]		# list that contains values of M_hd to try
M_list_ld = [10,20,30,40,50,60,70,80,90,100,200,300,400,416]					# list that contains values of M-ld to try

cols_hd = len(M_list_hd)
cols_ld = len(M_list_ld)

sample_list = [0,2,3,4,5,6]
rows = len(sample_list)

def reconstruct_image_HD_PCA(rows,cols,X_train,Y_train,A,eigenvectors,sample_list,M_list,average_training_face,main_title):
	index = 1
	font_size = 10
	count = 1
	plt.figure(figsize=(20,10))
	for sample in sample_list:
		
		face_sample = X_train[sample]							# using training_face[sample] as the training sample
		face_sample = np.array(face_sample)[np.newaxis]
		
		plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(face_sample,(46,56)).T, cmap = 'gist_gray')

		title = str("F"+str(Y_train[sample])+"-Original")
		plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
		index=index+1
		
		phi = x_train[:,sample]-average_training_face
		phi = np.array(phi)
		
		for M in M_list:
			reconstructed_image = average_training_face
			for i in range(0,M):
				print(main_title, ": F",Y_train[sample], "[", count, "/", rows, "] with M =", M, "... ( M = ", i, "/", M, ")    ", end="\r")

				u = np.array(eigenvectors[i].real)[np.newaxis]
				a = np.dot(u, phi.T)
				au = a*u
				reconstructed_image = reconstructed_image + au
				
			title = str("F"+str(Y_train[sample])+" [M=" + str(M) + "]")
			plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(reconstructed_image,(46,56)).T, cmap = 'gist_gray')
			plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
			index = index+1
			
			reconstructed_image = 0
			M = 0
		count = count+1
	plt.suptitle(main_title)
	plt.savefig(main_title)
	#plt.show()
	plt.close()
	
def reconstruct_image_LD_PCA(rows,cols,X_train,Y_train,A,eigenvectors,sample_list,M_list,average_training_face,main_title):
	index = 1
	font_size = 10
	count = 1
	plt.figure(figsize=(20,10))
	for sample in sample_list:
		
		face_sample = X_train[sample]							# using training_face[sample] as the training sample
		face_sample = np.array(face_sample)[np.newaxis]
		
		plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(face_sample,(46,56)).T, cmap = 'gist_gray')

		title = str("F"+str(Y_train[sample])+"-Original")
		plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
		index=index+1
		
		phi = x_train[:,sample]-average_training_face
		phi = np.array(phi)
		
		for M in M_list:
			reconstructed_image = average_training_face
			for i in range(0,M):
				print(main_title, ": F",Y_train[sample], "[", count, "/", rows, "] with M =", M, "... ( M = ", i, "/", M, ")    ", end="\r")

				v = np.array(eigenvectors[i].real)[np.newaxis]
				u = np.dot(v,A)
				u = u/norm(u)
				a = np.dot(u, phi.T)
				au = a*u
				reconstructed_image = reconstructed_image + au
				
			title = str("F"+str(Y_train[sample])+" [M=" + str(M) + "]")
			plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(reconstructed_image,(46,56)).T, cmap = 'gist_gray')
			plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
			index = index+1
			
			reconstructed_image = 0
			M = 0
		count = count+1
	plt.suptitle(main_title)
	plt.savefig(main_title)
	#plt.show()
	plt.close()
	
reconstruct_image_HD_PCA(rows,cols_hd,X_train,Y_train,A,eigenvectors_hd,sample_list,M_list_hd,average_training_face,"High-Dim PCA Reconstruction")
print("\n")
reconstruct_image_LD_PCA(rows,cols_ld,X_train,Y_train,A,eigenvectors_ld,sample_list,M_list_ld,average_training_face,"Low-Dim PCA Reconstruction")
