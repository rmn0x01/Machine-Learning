#Impor Library yang dibutuhkan
import pandas as pd
import math
import matplotlib.pyplot as plt

#Fungsi Prediksi, mengembalikan nilai Aktivasi
def predict(row,theta,bias):
    result = bias
    for i in range(len(row)-1):
        result += theta[i]*float(row[i])
    activation = 1/(1+math.exp(-result))
    return activation

#Fungsi untuk Training data
def train_data(train,learning_rate,theta,bias):
    #untuk menyimpan theta baru setelah di-update
    next_theta=theta
    #untuk menyimpan bias baru setelah di-update
    next_bias=bias
    sum_error = 0.0
    #Counter untuk menghitung seberapa banyak True Positive dan True Negative
    true_counter=0
    #List yang akan diisi dengan theta, bias, rata-rata error, dan akurasi
    returned=[]
    for row in train:
        #Hitung nilai aktivasi menggunakan fungsi sebelumnya
        activation = predict(row,next_theta,next_bias)
        prediction = 1.0 if activation>=0.5 else 0.0
        #Jika nilai prediksi sama dengan y, maka termasuk True Positive atau True Negative
        if(prediction == row[-1]):
            true_counter+=1
        
        for i in range(len(row)-1):
            dtheta = 2*(activation-row[-1])*(1-activation)*activation*row[i]
            next_theta[i] = next_theta[i]-learning_rate*dtheta
        
        dbias = 2*(activation-row[-1])*(1-activation)*activation
        next_bias=next_bias-learning_rate*dbias
        error = math.pow(activation-row[-1],2)
        sum_error += error
    
    returned.append(next_theta)
    returned.append(next_bias)
    returned.append(sum_error/len(train))
    returned.append(true_counter/len(train))
    return returned

#Fungsi Validasi
def proses_validasi(dataset,learning_rate,theta,bias):
    sum_error = 0.0
    #Menghitung true negative dan true positive
    counter=0
    #Mengembalikan rata-rata error dan akurasi
    returned=[]

    for row in dataset:
        activation = predict(row,theta,bias)
        prediction = 1.0 if activation>=0.5 else 0.0
        if(prediction == row[-1]):
            counter+=1
        error = math.pow(activation-row[-1],2)
        sum_error += error
    
    returned.append(sum_error/len(dataset))
    returned.append(counter/len(dataset))
    return returned

#Import dokumen, ubah nama kolom, ambil 100 data teratas
df = pd.read_csv('iris.csv')
df = df.rename(index=str,columns={"sepal_length":"x1","sepal_width":"x2","petal_length":"x3","petal_width":"x4"})
df.drop(df.index[-50:],inplace=True)
df['y']=1
#Drop kolom species untuk mempermudah proses berikutnya
df.loc[df['species'] == 'setosa','y'] = 0
df = df.drop(columns=['species'])
# 0 = setosa, 1 = versicolor

#Memecah (split) dataframe ke dalam masing-masing validasi dan training sesuai deskripsi yang telah dijelaskan sebelumnya
#Masing-masing validasi berisi 20 data, dimana 10 data dari setosa dan 10 data dari versicolor
validasi=[]
validasi.append(pd.concat([df.iloc[40:50], df.iloc[90:]]))
validasi.append(pd.concat([df.iloc[:10], df.iloc[50:60]]))
validasi.append(pd.concat([df.iloc[10:20], df.iloc[60:70]]))
validasi.append(pd.concat([df.iloc[20:30], df.iloc[70:80]]))
validasi.append(pd.concat([df.iloc[30:40], df.iloc[80:90]]))

#Masing-masing training berisi 80 data, diambil dari dataframe awal dikurangi data yang sudah menjadi validasi
training=[]
training.append(pd.concat([validasi[1],validasi[2],validasi[3],validasi[4]]))
training.append(pd.concat([validasi[0],validasi[2],validasi[3],validasi[4]]))
training.append(pd.concat([validasi[0],validasi[1],validasi[3],validasi[4]]))
training.append(pd.concat([validasi[0],validasi[1],validasi[2],validasi[4]]))
training.append(pd.concat([validasi[0],validasi[1],validasi[2],validasi[3]]))

#pemberian value awal
bias = 0.5
theta=[0.6,0.6,0.6,0.6]
learning_rate=0.1 #dan 0.8, cukup edit di line ini untuk mengubah learning_rate
n_epoch=300 #jumlah epoch

#List untuk menyimpan rata-rata error dan rata-rata akurasi dari training maupun validasi, per epoch per k-fold
list_avg_avg_error=[]
list_avg_accuracy=[]
list_avg_avg_error_v=[]
list_avg_accuracy_v=[]
#List untuk menyimpan update theta dan bias untuk tiap K-fold per epoch
list_theta=[theta for i in range(5)]
list_bias=[bias for i in range(5)]

for epoch in range(n_epoch):
    tmp_error = 0.0         #menyimpan error training
    tmp_accuracy = 0.0
    tmp_error_v = 0.0       #v = validasi
    tmp_accuracy_v = 0.0
    print(epoch) #debug untuk cek iterasi epoch ke-

    for i in range(len(training)):
        #bagian training
        y = train_data(training[i].head(80).values.tolist(),learning_rate,list_theta[i],list_bias[i])
        list_theta[i] = y[0]
        list_bias[i] = y[1]
        tmp_error+=y[2]
        tmp_accuracy+=y[3]
        #bagian validasi
        z = proses_validasi(validasi[i].head(20).values.tolist(),learning_rate,list_theta[i],list_bias[i])
        tmp_error_v+=z[0]
        tmp_accuracy_v+=z[1]
    
    list_avg_avg_error.append(tmp_error/len(training)*1.0)
    list_avg_accuracy.append(tmp_accuracy/len(training)*1.0)
    list_avg_avg_error_v.append(tmp_error_v/len(training)*1.0)
    list_avg_accuracy_v.append(tmp_accuracy_v/len(training)*1.0)

def cetak_grafik_loss_function(list_avg_avg_error,list_avg_avg_error_v):
    plt.plot(list_avg_avg_error,label='Training')
    plt.plot(list_avg_avg_error_v,label='Validation')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

def cetak_grafik_accuracy(list_avg_accuracy,list_avg_accuracy_v):
    plt.plot(list_avg_accuracy,label='Training')
    plt.plot(list_avg_accuracy_v,label='Validation')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


#Uncomment salah satu untuk menampilkan grafik
#cetak_grafik_loss_function(list_avg_avg_error,list_avg_avg_error_v)
#cetak_grafik_accuracy(list_avg_accuracy,list_avg_accuracy_v)
