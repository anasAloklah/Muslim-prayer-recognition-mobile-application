
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

import numpy as np


dataS="0.043086052;0.7467132;9.859121|0.020360291;0.73462725;9.874607|0.047271118;0.7451713;9.900895|0.024880545;0.72695625;9.942133|0.043766007;0.74341875;9.9259|0.030147787;0.7493468;9.915998|0.024009055;0.75040025;9.902475|0.03369121;0.72533774;9.894125|0.027131094;0.7492702;9.88539|0.01697968;0.74413705;9.86028|0.028864495;0.73942524;9.850148|0.03506069;0.74017227;9.865146|0.021452047;0.747058;9.841941|0.029831752;0.7431123;9.859676|0.017353173;0.7443765;9.852944|0.022476764;0.7422791;9.875393|0.02207454;0.7361404;9.847389|0.031996112;0.7488584;9.855214|0.034984075;0.7283449;9.861841|0.03196738;0.73364085;9.856354|0.028969841;0.7357957;9.8474|0.0267576;0.73412925;9.866783|0.027092787;0.7494617;9.867443|0.0424061;0.74043083;9.855349|0.032369606;0.7289769;9.877586|0.01934515;0.7332003;9.860079|0.00998861;0.7283449;9.860941|0.029132647;0.73404306;9.86029|0.007182605;0.7338132;9.851364|0.032905906;0.7386783;9.853271|0.021547815;0.7454778;9.862521|0.018454505;0.73038477;9.873679|0.032522835;0.71865314;9.845206|0.047520116;0.7235278;9.87568|0.036276944;0.7386208;9.857695|0.020685904;0.7460045;9.875526|0.029697677;0.73061454;9.854707|0.029257145;0.7290057;9.869216|0.030550014;0.7364852;9.848194|0.024985889;0.7324246;9.857369|0.030281862;0.7335929;9.835477|0.039130833;0.74314106;9.852648|0.023836672;0.72799057;9.8521595|0.018732235;0.72377676;9.853777|0.015830461;0.7408139;9.860692|0.004376601;0.730289;9.876082|0.027370514;0.7358148;9.8738785|0.035874717;0.7412544;9.875622|0.029745562;0.7357095;9.866774|0.014192827;0.72977185;9.861611|0.008446744;0.7324725;9.873697|0.0283665;0.72616136;9.861535|0.017190369;0.7456693;9.870595|0.020456059;0.73323864;9.862675|0.026048914;0.7274638;9.85825|0.012727576;0.7420876;9.861478|0.027964275;0.73682994;9.856325|0.009634268;0.7407756;9.856794|0.025771188;0.73903257;9.856268|0.044015;0.73325783;9.84648|0.0119039705;0.7243705;9.855789|0.021164743;0.735968;9.836099|0.0089543145;0.72597945;9.85394|0.021423317;0.74246114;9.86776|0.021021092;0.73344934;9.8910885|0.024679432;0.7305859;9.921716|0.015150508;0.7370885;9.926647|0.028663384;0.7463014;9.926351|0.02387498;0.7409671;9.913489|0.017602172;0.7297048;9.893818|0.042453982;0.73573816;9.885937|0.015945382;0.74675155;9.882057|0.032206804;0.73137116;9.852198|0.03331771;0.732032;9.864398|0.021729775;0.73377496;9.85055|0.019364303;0.7293025;9.8817425|0.028174965;0.72006094;9.862876|0.025158271;0.7273202;9.855233|0.030004134;0.7329705;9.864657|0.030731974;0.73041344;9.847754|"

dataS = dataS.rstrip('|')
rows = dataS.split('|')
arr = []
for row in rows:
    value = row.split(';')
    v = []
    v.append(float(value[0]))
    v.append(float(value[1]))
    v.append(float(value[2]))
    arr.append(v)

freamArry=arr
list=[]
list.append(freamArry)
freamArry=np.asarray(list).reshape(1, 80, 3, 1)
model = Sequential()
model.add(Conv2D(16, (2, 2), activation='relu', input_shape=freamArry[0].shape))
model.add(Dropout(0.02))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(4, activation='softmax'))
model.load_weights('asar_weights.h5')

# path_img = '1.jpg'
# res = requests.get(path_img) #to test the request
# test_image = Image.open(BytesIO('1.jpg'))


prediction = model.predict_classes(freamArry)  # predict the result
print(prediction)
res = ''
if prediction == [0]:
    res = 'Bowing'
elif prediction == [1]:
    res = 'prostration'
elif prediction == [2]:
    res = 'Sitting'
elif prediction == [3]:
    res = 'standing'
print(res)