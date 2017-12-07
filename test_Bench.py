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

def main(ind):
    train_Data = Qua2Eul(getData(0, 0)[:, ind]).transpose(1, 0)
    test_Data = Qua2Eul(getData(1, 0)[:, ind]).transpose(1, 0)
    regressor = tf.contrib.learn.Estimator(model_fn=lstm_model)
    train_X, train_y = generate_data(train_Data)
    test_X, test_y = generate_data(test_Data)
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

    plt.show()


main([2,3,4,5])