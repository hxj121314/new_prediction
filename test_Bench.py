from matplotlib import pyplot as plt
from Dataprocess.get_Data import *
from LSTM.lstm_Model import *
from Dataprocess.trans_Data import *
from result_Evaulation import *



def generate_data(seq):
    X = []
    y = []
    for i in range(seq.shape[-1] - TIMESTEPS - PREDICTSTEPS - 1):
        X.append(seq[:,i: i + TIMESTEPS])
        y.append(seq[:,i + TIMESTEPS + PREDICTSTEPS])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def main(video_id,ind,train_usrs,test_usrs):
    train_Data = Qua2Eul(getData(train_usrs[0], video_id)[:, ind]).transpose(1, 0)
    train_X, train_y = generate_data(train_Data)
    for i in train_usrs[1:-1]:
        train_Data_add = Qua2Eul(getData(i, video_id)[:, ind]).transpose(1, 0)
        train_X_add, train_y_add = generate_data(train_Data_add)
        train_X = np.vstack((train_X,train_X_add))
        train_y = np.vstack((train_y,train_y_add))
    test_Data = Qua2Eul(getData(test_usrs[0], video_id)[:, ind]).transpose(1, 0)
    test_X, test_y = generate_data(test_Data)
    for j in test_usrs[1:-1]:
        test_Data_add = Qua2Eul(getData(j, video_id)[:, ind]).transpose(1, 0)
        test_X_add, test_y_add = generate_data(test_Data_add)
        test_X = np.vstack((test_X,test_X_add))
        test_y = np.vstack((test_y,test_y_add))
    regressor = tf.contrib.learn.Estimator(model_fn=lstm_model)
    regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
    predicted = []
    for pred in regressor.predict(test_X):
        predicted.append(pred)
    predicted = np.array(predicted, dtype=np.float32)


    plt.subplot(1,2,1)
    plt.plot(test_y)
    plt.subplot(1,2,2)
    plt.plot(predicted)
    acu, mean = dif_Ang(predicted, test_y)
    print(acu,'Acu')
    print(mean,'Mean')

    #plt.show()
    plt.savefig('/home/hxj/new_prediction/fig.png')

# ind = [2,3,4,5]
# print(getData(11, 0))
# test_Data = Qua2Eul(getData(11, 0)[:, ind]).transpose(1, 0)
# test_X, test_y = generate_data(test_Data)
# test_Data_add = Qua2Eul(getData(2, 0)[:, ind]).transpose(1, 0)
# test_X_add, test_y_add = generate_data(test_Data_add)
# print(test_X.shape)
# print(test_X_add.shape)
# print(np.vstack((test_X,test_X_add)).shape)

# plt.plot(test_y[:,1])
# plt.show()
# print(test_y)
# acu, mean = dif_Ang(test_y, test_y)
main(2,[2,3,4,5],np.arange(0,49,1),np.arange(50,59,1))