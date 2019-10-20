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


# In[ ]:




