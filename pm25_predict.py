import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
name = "aqa_montesori_nivel_calle"
sumy = 0
mesures = pd.read_csv( name+".csv")

epochs = mesures["time"].values
#pm25s = mesures[name+".mean_pm25"].values
pm25s = mesures[name+".pm25"].values
lr = linear_model.LinearRegression()
#pm25s = [pm25 for pm25 in pm25s]
epochs = [epoch for epoch in range(len(epochs))]
epochs = np.asanyarray(epochs)
pm25s = np.asanyarray(pm25s)

epochs = epochs.reshape((1,-1))
pm25s = pm25s.reshape((1,-1))

lr.fit(epochs,pm25s )
print(epochs)

for i in epochs:
    print(i,")",lr.predict(pm25s) )
    print(i,")",pm25s)
    #print("-----"*25)
print(pm25s)
print("-----"*25)
y = lr.predict(pm25s)
print(y)
for i in y:
    sumy +=y
b = sumy/len(epochs)
for i in range (len(epochs)):
    w = ((epochs[i]*epochs)-(pm25s[i]*pm25s))/((epochs[i]*epochs)**2)
prediction = w*epochs+b
#sumation=1/len(epochs)*sum(y-prediction*(epochs)**2,len(epochs))
#print(sumation)
for x in epochs:
    for y2 in y:
        filtred = np.polyfit(x, y2, 1)
print('Fitted Parameters:', filtred)

predictions = np.polyval(filtred, epochs)
absError = predictions - y

SE = np.square(absError) # squared errors
MSE = np.mean(SE) # mean squared errors
RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (np.var(absError) / np.var(y))
plt.xlabel("epochs")
plt.ylabel("pm25")
plt.title("unloquer predict")
plt.plot(epochs,pm25s,'bo')
#plt.plot(epochs,,'go')
plt.plot(epochs,y,'g-')
plt.plot(epochs,prediction,'b-')
#X = np.linspace(min(epochs), max(epochs))
Y = np.polyval(predictions, epochs)
plt.plot(epochs*2,Y,"go",linewidth=2.0)
#plt.plot(epochs,prediction ,'ro',linewidth=2.0)
plt.show()
