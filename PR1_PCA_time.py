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
dir = os.getcwd()						# gets the working directory of the file this python script is in
os.chdir (dir)							# changes the working directory to the file this python script is in
										# also change plt.savefig to plt.show
										
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
print(eigenvectors_hd.real, "\n")
print("Low-Dimension Eigenvectors = ")
print(eigenvectors_ld.real, "\n")

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

from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#### FACE IMAGE RECONSTRUCTION AND TIME TAKEN
X_train = x_train.squeeze()				# This is the matrix that contains all of the face training data
Y_train = y_train						# This is the matrix that contains all of the labels for the training data
training_face0 = X_train[0]
training_face0 = np.array(training_face0)[np.newaxis]

print("A Training Face has shape: ", training_face0.shape)
print("Average Training Face has shape: ", average_training_face.shape, "\n")

def plot_time(df_time_hd, df_time_ld):		# function for plotting high-dimension and low-dimension eigenvalues
	time_hd = df_time_hd['Time Taken (HD-PCA)']
	M_hd = df_time_hd['M']
	time_ld = df_time_ld['Time Taken (LD-PCA)']
	M_ld = df_time_ld['M']

	plt.figure(figsize=(8,8))
	plt.scatter(M_hd, time_hd, color = 'red', s=50, alpha=1, marker = 'o')	# creating a scatter plot for the high-dim eigenvalues
	plt.scatter(M_ld, time_ld, color = 'blue', s=50, alpha=1, marker = 'x')	# creating a scatter plot for the low-dim eigenvalues
	plt.plot(M_hd, time_hd, color = 'red')									# creating line graph for high-dim eigenvalues
	plt.plot(M_ld, time_ld, color = 'blue')									# creating line graph for lower-dim eigenvalues

	plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)				# parameters for plot grid
	plt.xticks(np.arange(0,max(M_hd)+50, 50))								# adjusting the intervals to 250
	#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))			# setting y-axis to scientific scale
	plt.title('Comparing High-Dimension and Low-Dimension PCA Reconstruction Time').set_position([0.5,1.05])

	plt.xlabel('M')
	plt.ylabel('Time Taken(s)')			
	plt.legend(loc = 'upper left')											# creating legend and placing in at the top right
	#plt.savefig('Comparing High-Dimension and Low-Dimension PCA Reconstruction Time')
	plt.show()
	plt.close()

def time_taken_to_reconstructan_an_image_PCA(A,eigenvectors,average_training_face,M_list):
	sample = 0
	M_array = []
	M_time_array = []
	count = 1
	rows = len(M_list)
	for M in M_list:
		start_time = time.time()
		reconstructed_image = average_training_face
		for i in range(0,M):
			print("F",Y_train[sample], "[", count, "/", rows, "] with M =", M, "... ( M = ", i+1, "/", M, ")    ", end="\r")
			phi = np.array(A[sample])[np.newaxis]	
			u = np.array(eigenvectors[i].real)[np.newaxis]
			a = np.dot(phi.T, u)
			au = np.dot(a,u.T)
			reconstructed_image = reconstructed_image + au.T

		M_time = time.time() - start_time
		M_array.append(M)
		M_time_array.append(M_time)
		
		reconstructed_image = 0
		M = 0
		count = count+1
	return M_array, M_time_array

M_list_hd_time = [1,10,50,100,200,300,400,416]						# list that contains values of M_hd to try to get time
M_list_ld_time = [1,10,50,100,200,300,400,416]						# list that contains values of M_ld to try to get time
M_array_hd, M_time_array_hd = time_taken_to_reconstructan_an_image_PCA(A,eigenvectors_hd,average_training_face,M_list_hd_time)
print("\n")
M_array_ld, M_time_array_ld = time_taken_to_reconstructan_an_image_PCA(A,eigenvectors_ld,average_training_face,M_list_ld_time)

#### CONVERT TIME TAKEN INTO DATAFRAME TO PLOT 
M_array_hd = np.array(M_array_hd)									# converting into a numpy array
M_time_array_hd = np.array(M_time_array_hd)
M_array_ld = np.array(M_array_ld)								
M_time_array_ld = np.array(M_time_array_ld)

df_time_hd = DataFrame(M_array_hd, columns = ['M'])					# converting high-dim array into a dataframe
df_time_hd['Time Taken (HD-PCA)'] = M_time_array_hd
df_time_ld = DataFrame(M_array_ld, columns = ['M'])							
df_time_ld['Time Taken (LD-PCA)'] = M_time_array_ld

combined_time_df = pd.concat([df_time_hd, df_time_ld],axis = 1)		# combining the high-dim and low-dim dataframes into one

plot_time(df_time_hd, df_time_ld)

print("Time taken to obtain High-Dim Covariance Matrix: ", A_time+covariance_matrix_hd_time, "s")
print("Time taken to obtain Low-Dim Covariance Matrix: ", A_time+covariance_matrix_ld_time, "s \n")

print("Time taken to obtain High-Dim Eigenvalues and Eigenvectors: ", eigenvalues_hd_eigenvectors_hd_time, "s")
print("Time taken to obtain Low-Dim Eigenvalues and Eigenvectors: ", eigenvalues_ld_eigenvectors_ld_time, "s \n")

sys.exit()