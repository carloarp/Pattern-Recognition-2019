#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
mat_content = sio.loadmat('face.mat')

mat_content


# In[2]:


face_data = mat_content['X']

print(face_data) # Each column represents one face image, each row a pixel value for a particular coordinate of the image
print(face_data.shape) #(D, N), dimension x cardinality


# In[3]:


face_157 = face_data[:,157]

print(face_157.shape)
print(face_157)


# In[4]:


# write a function that converts s D-dimensional vector to 46x56 pixels 2D image

face_157 = np.reshape(face_157,(46,56))
print(face_157)



# In[5]:


plt.imshow(face_157, cmap = 'gist_gray')


# In[6]:


face_157 = face_157.T
plt.imshow(face_157,cmap = 'gist_gray')


# In[7]:


print(mat_content)


# In[8]:


face_labels = mat_content['l']
print(face_labels[0,157])


# In[9]:


face_1 = face_data[:,5]
face_2 = face_data[:,180]

avg_face = (0.5*face_1 + 0.5*face_2)

plt.subplot(311),plt.imshow(np.reshape(face_1,(46,56)).T, cmap = 'gist_gray')
plt.title('Face 1'), plt.xticks([]), plt.yticks([])
plt.subplot(312),plt.imshow(np.reshape(face_2,(46,56)).T, cmap = 'gist_gray')
plt.title('Face 2'), plt.xticks([]), plt.yticks([])
plt.subplot(313),plt.imshow(np.reshape(avg_face,(46,56)).T, cmap = 'gist_gray')
plt.title('Face avg'), plt.xticks([]), plt.yticks([])

print(avg_face)
plt.show()


# In[12]:


face_label_array = []
face_data_array = []
for i in range(0,520):
    face_label_array.append(face_labels[0,i]) 
    face_data_array.append(face_data[:,i])
    


# In[13]:


print(face_label_array)
print(face_data_array)


# In[16]:


data_train, data_test, label_train, label_test = train_test_split(face_data_array, face_label_array, random_state = 42) #splitting into training data and test data


# In[17]:


print(data_train)


# In[18]:


print(len(data_train))


# In[19]:


print(len(data_test))


# In[20]:


sum_data_train = np.zeros(2576)
for i in range(0, len(data_train)):
    sum_data_train += data_train[i] 
    face_mean = sum_data_train/len(data_train)
print(face_mean)
    


# In[21]:


face_mean_pixel = np.reshape(face_mean,(46,56))






# In[22]:


plt.imshow(face_mean_pixel, cmap = 'gist_gray')


# In[23]:


face_mean_pixel = face_mean_pixel.T
plt.imshow(face_mean_pixel, cmap = 'gist_gray')


# 

# In[95]:


#print(data_train[0] - face_mean)
S_ih = np.zeros((2576, 2576))
for i in range(0, len(data_train)):
    phi =  data_train[i] - face_mean
    S_ih += phi*(phi.T)
S_fh = S_ih /len(data_train)
print(S_fh.shape)
#print(data_train[0])
#print(data_train[0].T)
#print(data_train[0] -face_mean)
#print((data_train[0]-face_mean).shape)
    


# In[25]:


#eigenvalue decomposition of S


# In[108]:


from scipy.linalg import eigh
eigenval, eigenvec = eigh(S_fh, eigvals = (2476, 2575)) #obtain 1000th largest eigenvalues, arrange in ascending order
print(eigenval.shape)

print(eigenvec.shape)


# In[109]:


print(eigenval)


# In[51]:


#top 1000 eigenfaces

plt.subplot(511),plt.imshow(np.reshape(eigenvec[:,990],(46,56)).T, cmap = 'gist_gray')
plt.title('Face 1'), plt.xticks([]), plt.yticks([])
plt.subplot(512),plt.imshow(np.reshape(eigenvec[:,991],(46,56)).T, cmap = 'gist_gray')
plt.title('Face 2'), plt.xticks([]), plt.yticks([])
plt.subplot(513),plt.imshow(np.reshape(eigenvec[:,992],(46,56)).T, cmap = 'gist_gray')
plt.title('Face 3'), plt.xticks([]), plt.yticks([])
plt.subplot(514),plt.imshow(np.reshape(eigenvec[:,993],(46,56)).T, cmap = 'gist_gray')
plt.title('Face 4'), plt.xticks([]), plt.yticks([])
plt.subplot(515),plt.imshow(np.reshape(eigenvec[:,999],(46,56)).T, cmap = 'gist_gray')
plt.title('Face 5'), plt.xticks([]), plt.yticks([])
#plt.subplot(10,1,1),plt.imshow(np.reshape(eigenvec[:,995],(46,56)).T, cmap = 'gist_gray')
#plt.title('Face 6'), plt.xticks([]), plt.yticks([])
#plt.subplot(10,1,1),plt.imshow(np.reshape(eigenvec[:,996],(46,56)).T, cmap = 'gist_gray')
#plt.title('Face 7'), plt.xticks([]), plt.yticks([])
#plt.subplot(10,1,1),plt.imshow(np.reshape(eigenvec[:,997],(46,56)).T, cmap = 'gist_gray')
#plt.title('Face 8'), plt.xticks([]), plt.yticks([])
#plt.subplot(10,1,1),plt.imshow(np.reshape(eigenvec[:,998],(46,56)).T, cmap = 'gist_gray')
#plt.title('Face 9'), plt.xticks([]), plt.yticks([])
#plt.subplot(10,1,1),plt.imshow(np.reshape(eigenvec[:,999],(46,56)).T, cmap = 'gist_gray')
#plt.title('Face 10'), plt.xticks([]), plt.yticks([])


# In[52]:


plt.subplot(5, 1, 1),plt.imshow(np.reshape(eigenvec[:,999],(46,56)).T, cmap = 'gist_gray')


# In[54]:


plt.subplot(5, 2, (1,2)),plt.imshow(np.reshape(eigenvec[:,999],(46,56)).T, cmap = 'gist_gray')


# In[55]:


plt.subplot(5, 2, (1,2)),plt.imshow(np.reshape(eigenvec[:,999],(46,56)).T, cmap = 'gist_gray')


# In[80]:


plt.subplot(2, 10,1),plt.imshow(np.reshape(eigenvec[:,999],(46,56)).T, cmap = 'gist_gray')
plt.subplot(2, 10,10),plt.imshow(np.reshape(eigenvec[:,998],(46,56)).T, cmap = 'gist_gray')
plt.subplot(2, 10,20),plt.imshow(np.reshape(eigenvec[:,999],(46,56)).T, cmap = 'gist_gray')
#plt.subplot(2, 5,(1, 3)),plt.imshow(np.reshape(eigenvec[:,997],(46,56)).T, cmap = 'gist_gray')
#plt.subplot(2, 5,(1, 4)),plt.imshow(np.reshape(eigenvec[:,996],(46,56)).T, cmap = 'gist_gray')
#plt.subplot(2, 5,(1, 5)),plt.imshow(np.reshape(eigenvec[:,995],(46,56)).T, cmap = 'gist_gray')
#plt.subplot(2, 5,(2, 1)),plt.imshow(np.reshape(eigenvec[:,994],(46,56)).T, cmap = 'gist_gray')
#plt.subplot(2, 5,(2, 2)),plt.imshow(np.reshape(eigenvec[:,993],(46,56)).T, cmap = 'gist_gray')
#plt.subplot(2, 5,(2, 3)),plt.imshow(np.reshape(eigenvec[:,992],(46,56)).T, cmap = 'gist_gray')
#plt.subplot(2, 5,(2, 4)),plt.imshow(np.reshape(eigenvec[:,991],(46,56)).T, cmap = 'gist_gray')
#plt.subplot(2, 5,(2, 5)),plt.imshow(np.reshape(eigenvec[:,990],(46,56)).T, cmap = 'gist_gray')


# In[88]:


iteration = 1
for i in range(2):
    for j in range(5):
        plt.subplot((i+1),(j+1),(iteration),plt.imshow(np.reshape(eigenvec[:,(1000-iteration)],(46,56)).T, cmap = 'gist_gray')
        iteration += 1


# In[92]:


plt.subplot(2, 5,1),plt.imshow(np.reshape(eigenvec[:,999],(46,56)).T, cmap = 'gist_gray')
plt.subplot(2, 5,2),plt.imshow(np.reshape(eigenvec[:,998],(46,56)).T, cmap = 'gist_gray')
plt.subplot(2, 5,3),plt.imshow(np.reshape(eigenvec[:,997],(46,56)).T, cmap = 'gist_gray')
plt.subplot(2, 5,4),plt.imshow(np.reshape(eigenvec[:,996],(46,56)).T, cmap = 'gist_gray')
plt.subplot(2, 5,5),plt.imshow(np.reshape(eigenvec[:,995],(46,56)).T, cmap = 'gist_gray')
plt.subplot(2, 5,6),plt.imshow(np.reshape(eigenvec[:,994],(46,56)).T, cmap = 'gist_gray')
plt.subplot(2, 5,7),plt.imshow(np.reshape(eigenvec[:,993],(46,56)).T, cmap = 'gist_gray')
plt.subplot(2, 5,8),plt.imshow(np.reshape(eigenvec[:,992],(46,56)).T, cmap = 'gist_gray')
plt.subplot(2, 5,9),plt.imshow(np.reshape(eigenvec[:,991],(46,56)).T, cmap = 'gist_gray')
plt.subplot(2, 5,10),plt.imshow(np.reshape(eigenvec[:,990],(46,56)).T, cmap = 'gist_gray')


# In[103]:


#phi =  data_train[0] - face_mean
#print((phi.T)*phi) 

S_il = np.zeros((390, 390))
for i in range(0, 390): #row
    for j in range(0, 390): #columm
        phi_row =  data_train[i] - face_mean
        phi_column = data_train[j] - face_mean
        S_il[i,j] = np.dot(phi_column, phi_row)
S_fl = S_il /len(data_train)
print(S_fl.shape)


# In[107]:


from scipy.linalg import eigh
eigenval_l, eigenvec_l = eigh(S_fl, eigvals = (290, 389)) #obtain 1000th largest eigenvalues, arrange in ascending order
print(eigenval_l.shape)
print(eigenval_l)
print(eigenvec_l.shape)


# In[3]:


import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
#import cv2
#import sys
#import time
#import os

from scipy.io import loadmat
from sklearn.decomposition import PCA
										# comment out this code when running on jupyter
#dir = os.getcwd()						# gets the working directory of the file this python script is in
#os.chdir (dir)							# changes the working directory to the file this python script is in
										# also change plt.savefig to plt.show
										
def print_image(face_image):			# function for plotting an image
	face_image = np.reshape(face_image, (46,56))
	face_image = face_image.T
	plt.imshow(face_image, cmap='gist_gray')
	plt.show()

#def save_image(face_image,title):		# function for saving an image
#	face_image = np.reshape(face_image, (46,56))
#	face_image = face_image.T
#	plt.imshow(face_image, cmap='gist_gray')
#	plt.title(title)
#	plt.savefig(title)

mat_content = loadmat('face.mat')			# unpacks the .mat file

#print("\n")
#print("Showing contents of the .mat file:")		# shows the contents of the .mat file
#print(mat_content)

# Array 'X' contains the face data	
# Each column represents one face image
# Each row represents a pixel value for a particular coordinate of the image
face_data = mat_content['X']
face_labels = mat_content['l']	

#print("\n")
#print("Showing Face Data = mat_content['X']:")					
#print(face_data)

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


# In[10]:


from scipy.linalg import eigh
eigenvalues_hd, eigenvectors_hd = eigh(covariance_matrix_hd)	# obtain eigenvalues and eigenvectors from the high-dimension covariance matrix
eigenvalues_ld, eigenvectors_ld = eigh(covariance_matrix_ld)	# obtain eigenvalues and eigenvectors from the low-dimension covariance matrix

#### SORT EIGENVALUES AND EIGENVECTORS FROM HIGHEST TO LOWEST
idx_hd = eigenvalues_hd.argsort()[::-1]
eigenvalues_hd = eigenvalues_hd[idx_hd]
eigenvectors_hd = eigenvectors_hd[:,idx_hd]
print(idx_hd)

idx_ld = eigenvalues_ld.argsort()[::-1]
eigenvalues_ld = eigenvalues_ld[idx_ld]
eigenvectors_ld = eigenvectors_ld[:,idx_ld]
print(idx_ld)

print("High-Dimension eigenvalues = ")
print(eigenvalues_hd, "\n")
print(eigenvalues_hd.shape)
print("Low-Dimension eigenvalues = ")
print(eigenvalues_ld, "\n")

print("High-Dimension Eigenvectors = ")
print(eigenvectors_hd.real, "\n")
print("Low-Dimension Eigenvectors = ")
print(eigenvectors_ld.real, "\n")

print("Number of High-Dimension eigenvalues: ", eigenvalues_hd.shape[0])
print("Number of High-Dimension eigenvectors: ", eigenvectors_hd.shape, "\n")
print("Number of Low-Dimension eigenvalues: ", eigenvalues_ld.shape[0])
print("Number of Low-Dimension eigenvectors: ", eigenvectors_ld.shape, "\n")


# In[11]:



number_of_eigenvalues_hd = eigenvalues_hd.shape[0]
number_of_eigenvalues_ld = eigenvalues_ld.shape[0]
print(number_of_eigenvalues_hd)
print(number_of_eigenvalues_ld)


# In[14]:


print(eigenvalues_hd)
eigenvalues_hd_array = []
eigenvectors_hd_array = []
eigenvalues_hd_index_array = []
number_of_non_zero_eigenvalues_hd = 0

i = 0
while(eigenvalues_hd[i]!= 0):

    number_of_non_zero_eigenvalues_hd += 1
    eigenvalues_hd_array.append(eigenvalues_hd[i])
    eigenvectors_hd_array.append(eigenvectors_hd[i])
    eigenvalues_hd_index_array.append(i)
    i +=1

print("Number of non-zero High-Dimension Eigenvalues = ", number_of_non_zero_eigenvalues_hd)


# In[40]:


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


print("Number of non-negative and non-zero High-Dimension Eigenvalues = ", number_of_non_negative_eigenvalues_hd)



# In[18]:


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


# In[38]:


from numpy.linalg import matrix_rank
print(matrix_rank(covariance_matrix_hd))


# In[15]:


print("A HD eigenvector has shape ", np.array(eigenvectors_hd[0])[np.newaxis].shape)
print("A LD eigenvector has shape ", np.array(eigenvectors_ld[0])[np.newaxis].shape,"\n")


# In[23]:


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
				reconstructed_image += au
				
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


# In[19]:


reconstruct_image_HD_PCA(rows,cols_hd,X_train,Y_train,A,eigenvectors_hd,sample_list,M_list_hd,average_training_face,"High-Dim PCA Reconstruction")
print("\n")


# In[63]:


import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.decomposition import PCA
from numpy.linalg import norm

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

mat_content = loadmat('face.mat')			# unpacks the .mat file

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

A = A.T


covariance_matrix_hd = np.matmul(A,A.T)/train_size
covariance_matrix_ld = np.matmul(A.T,A)/train_size

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

print(eigenvectors_hd)
print(eigenvectors_ld)

print("A HD eigenvector has shape ", eigenvectors_hd.shape)
print("A LD eigenvector has shape ", eigenvectors_ld.shape,"\n")

print(" A HD eigenvector ", eigenvectors_hd[:,0])

u0 = np.dot(A, eigenvectors_ld[:,0])

print(u0/norm(u0))





# In[ ]:


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


# In[64]:


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
	plt.show()
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
	plt.show()
	plt.close()
	


# In[65]:


reconstruct_image_HD_PCA(rows,cols_hd,X_train,Y_train,A,eigenvectors_hd,sample_list,M_list_hd,average_training_face,"High-Dim PCA Reconstruction")


# In[83]:


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
		
		for M in M_list:
			reconstructed_image = average_training_face
			for i in range(0,M):
				v = eigenvectors[:,i]
				u = np.dot(A, v)
				print()
				u = u/norm(u)
				a = np.dot(u, phi.T)
				au = a*u
				reconstructed_image += au
			title = str("F"+str(Y_train[sample])+" [M=" + str(M) + "]")
			plt.subplot(rows+1,cols+1,index), plt.imshow(np.reshape(reconstructed_image,(46,56)).T, cmap = 'gist_gray')
			plt.title(title, fontsize=font_size).set_position([0.5,0.95]), plt.xticks([]), plt.yticks([])
			index = index+1
			
			reconstructed_image = 0
			M = 0
		count = count+1
	plt.suptitle(main_title)
	plt.savefig(main_title)
	plt.show()
	plt.close()


# In[84]:


reconstruct_image_LD_PCA(rows,cols_ld,X_train,Y_train,A,eigenvectors_ld,sample_list,M_list_ld,average_training_face,"Low-Dim PCA Reconstruction")


# In[ ]:




