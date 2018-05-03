import numpy as np
np.set_printoptions(threshold=np.inf)


path = "D:/"

data = np.loadtxt(path + 'testimgresult' + str(1) +'.csv', delimiter=" ")

for i in range(9):
    tmp = np.loadtxt(path + 'testimgresult' + str(i+2) +'.csv', delimiter=" ")
    data = np.vstack((data, tmp))

np.savetxt('D:/testimg/clusater.csv', data, delimiter = ',')