import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
name = "aqa_montesori_nivel_calle"
sumy = 0
mesures = pd.read_csv( name+".csv")
epoch = mesures["time"].values
pm25 = mesures[name+".pm25"].values
epoch = [epoch for epoch in range(len(epoch))]
epoch = np.asanyarray(epoch)
pm25 = np.asanyarray(pm25)
epochs = epoch.reshape((1,-1))
pm25s = pm25.reshape((1,-1))


y = pm25s 
for i in y:
    sumy +=y
b = sumy/len(epochs)
for i in range (len(epochs)):
    w = ((epochs[i]*epochs)-(pm25s[i]*pm25s))/((epochs[i]*epochs)**2)
prediction = w*epochs+b

for x in epochs:
    for y2 in y:
        filtred = np.polyfit(x, y2, 1)

predictions = np.polyval(filtred, epochs)

Y = np.polyval(predictions, epochs)

plt.xlabel("epochs")
plt.ylabel("pm25")
plt.title("unloquer predict")
plt.plot(epoch,pm25,'bo',label='pm25')
plt.plot(epochs*2,Y,"go",label='prediction')
plt.legend()#(points,"pm25"),(pred,"prediction") )

plt.show()
