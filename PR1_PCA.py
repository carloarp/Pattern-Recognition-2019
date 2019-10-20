### Finds the High and Low Dimension:
### 1. Covariance Matrix
### 2. Eigenvalues and Eigenvectors
### 3. Non-zero and Non-negative Eigenvectors and Eigenvalues
### 4. Compare High- and Low-Dimension Eigenvalues

### Importing depdendencies
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

#print_image(average_training_face)
#save_image(average_training_face,"Average Training Face")

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

print("Covariance Matrix (High-Dimension) has shape: ", covariance_matrix_hd.shape)
print("Covariance Matrix (Low-Dimension) has shape: ", covariance_matrix_ld.shape, "\n")

print("Covariance Matrix (High-Dimension) = ")
print(covariance_matrix_hd, "\n")
print("Covariance Matrix (Low-Dimension) = ")
print(covariance_matrix_ld, "\n")

#### EIGENVALUES AND EIGENVECTORS
eigenvalues_hd, eigenvectors_hd = np.linalg.eig(covariance_matrix_hd)	# obtain eigenvalues and eigenvectors from the high-dimension covariance matrix
eigenvalues_ld, eigenvectors_ld = np.linalg.eig(covariance_matrix_ld)	# obtain eigenvalues and eigenvectors from the low-dimension covariance matrix

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

def plot_eigenvalues(df_hd, df_ld):		# function for plotting high-dimension and low-dimension eigenvalues
	hd_index = df_hd['HD Index']
	hd_eigenvalues = df_hd['High-Dimension Eigenvalues']
	ld_index = df_ld['LD Index']
	ld_eigenvalues = df_ld['Low-Dimension Eigenvalues']

	plt.figure(figsize=(8,8))
	plt.scatter(hd_index, hd_eigenvalues, color = 'red', s=50, alpha=1, marker = 'o')	# creating a scatter plot for the high-dim eigenvalues
	plt.scatter(ld_index, ld_eigenvalues, color = 'blue', s=50, alpha=1, marker = 'x')	# creating a scatter plot for the low-dim eigenvalues
	plt.plot(hd_index, hd_eigenvalues, color = 'red')									# creating line graph for high-dim eigenvalues
	plt.plot(ld_index, ld_eigenvalues, color = 'blue')									# creating line graph for lower-dim eigenvalues

	plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)							# parameters for plot grid
	plt.xticks(np.arange(min(hd_index),max(hd_index)+250, 250))							# adjusting the intervals to 250
	plt.yticks(np.arange(min(hd_eigenvalues),max(hd_eigenvalues)+25000, 50000))			# adjusting the intervals to 50,000
	plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))						# setting y-axis to scientific scale
	plt.title('Comparing High-Dimension and Low-Dimension PCA Eigenvalues')

	plt.xlabel('Eigenvalue Index (M)')
	plt.ylabel('Eigenvalue')			
	plt.legend(loc = 'upper right')														# creating legend and placing in at the top right
	plt.show()	

plot_eigenvalues(df_hd, df_ld)
sys.exit()
