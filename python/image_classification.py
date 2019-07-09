import numpy as np 
import faiss
import time
import cv2
import glob

t1 = time.time()
features = np.loadtxt('test.txt')
features_shape = features.astype(np.float32)
t2 = time.time()
print('t1 = ',t2-t1)
# features_astype = features.astype(np.float32)

# mat = faiss.PCAMatrix (1024, 128)
# mat.train(features_astype)
# assert mat.is_trained
# features_shape = mat.apply_py(features_astype)
# print(features_shape.shape)
# np.savetxt('PCA_features.txt',features_shape)

print(features_shape.shape)

dimension = 1024
n = 95276
nlist = 50
quantiser = faiss.IndexFlatL2(dimension)  
index = faiss.IndexIVFFlat(quantiser, dimension, nlist,   faiss.METRIC_L2)

# print(index.is_trained)  
index.train(features_shape) 
# print(index.ntotal)  
index.add(features_shape)   
# print(index.is_trained)  
# print(index.ntotal)

nprobe = 10  # find 2 most similar clusters
k = 5  # return 3 nearest neighbours 

a = np.reshape(features_shape[1], (1, -1))
distances, indices = index.search(a, k)
print(distances)
print(indices)

files = glob.glob('/home/duc-dn/test/*/*.jpg')
for i in range(len(files)):
    indices[0].sort()
    for j in range(len(indices[0])):
        if i == indices[0][j]:
            img = cv2.imread(files[i])
            cv2.imshow('a'+str(i),img)
cv2.waitKey(0)

# for i in range(features_shape.shape[0]):
#     a = np.reshape(features_shape[i], (1, -1))
#     distances, indices = index.search(a, k)
#     print(distances)
#     print(indices)

print('t2 = ',time.time()-t2)