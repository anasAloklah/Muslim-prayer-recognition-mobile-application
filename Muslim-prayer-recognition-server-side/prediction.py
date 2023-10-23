from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from FSM import FSM_fajer,FSM_asar,FSM_magrab,simlarty_asar,simlarty_fajer,simlarty_magrab,remove_noise,remove_redance,corrctor
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
def predcition(data="",type=""):
    data = data.rstrip('|')
    rows = data.split('|')
    arr = []
    for row in rows:
        value = row.split(';')
        v = []
        v.append(float(value[0]))
        v.append(float(value[1]))
        v.append(float(value[2]))
        arr.append(v)

    freamArry = arr
    list = []
    list.append(freamArry)
    num_of_samples=int(len(freamArry)/80)
    scaler = StandardScaler()
    #freamArry = scaler.fit_transform(freamArry)
    freamArry = np.asarray(list).reshape(num_of_samples, 80, 3, 1)

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
    list_activties_prediction = model.predict_classes(freamArry)  # predict the activties of salat
    print(list_activties_prediction)
    if(type=="aser" or type=="zuher" or type=="esha" ):
        salat_asar=remove_noise(list_activties_prediction,6)
        salat_asar = remove_redance(salat_asar)
        #salat_asar = corrctor(salat_asar)
        valid_salat=""
        if(FSM_asar(salat_asar)==True):
            valid_salat="is valid"
        else:valid_salat="not valid"
        precent_correct=simlarty_asar(salat_asar)
        count_standing=salat_asar.count(3)#standing
        count_bowing = salat_asar.count(0)
        count_prostration= salat_asar.count(1)
        count_sitting= salat_asar.count(2)
        if(precent_correct>93):
            valid_salat="is valid"
        result='{"salat": "'+valid_salat+'", "precent_correct": "'+precent_correct+'", "standing":"'+count_standing+'"' \
                 ',"bowing":"'+count_bowing+'","prostration" : "'+count_prostration+'","sitting":"'+count_sitting+'" }'
        result_dict = json.loads(result)
        return result_dict

    if (type == "fajer" ):
        salat_fajer = remove_noise(list_activties_prediction,3)
        print(salat_fajer)
        salat_fajer = remove_redance(salat_fajer)
        #salat_fajer=corrctor(salat_fajer)
        print(salat_fajer)
        valid_salat = ""
        if (FSM_fajer(salat_fajer) == True):
            valid_salat = "is valid"
        else:
            valid_salat = "not valid"
        precent_correct = simlarty_fajer(salat_fajer)
        count_standing = salat_fajer.count(3)
        count_bowing = salat_fajer.count(0)
        count_prostration = salat_fajer.count(1)
        count_sitting = salat_fajer.count(2)
        if(precent_correct>93):
            valid_salat="is valid"
        result = '{"salat": "' + valid_salat + '", "precent_correct": "' + str(precent_correct) + '", "standing":"' + str(count_standing) + '"' \
                  ',"bowing":"' + str(count_bowing) + '","prostration" : "' + str(count_prostration) + '","sitting":"' + str(count_sitting) + '" }'
        result_dict = json.loads(result)
        return result_dict

    if (type == "magreb" ):
        salat_magram= remove_noise(list_activties_prediction,3)
        salat_magram = remove_redance(salat_magram)
        #salat_magram = corrctor(salat_magram)
        valid_salat = ""
        if (FSM_magrab(salat_magram) == True):
            valid_salat = "is valid"
        else:
            valid_salat = "not valid"
        precent_correct = simlarty_magrab(salat_magram)
        count_standing = salat_magram.count(3)
        count_bowing = salat_magram.count(0)
        count_prostration = salat_magram.count(1)
        count_sitting = salat_magram.count(2)
        if(precent_correct>93):
            valid_salat="is valid"
        result = '{"salat": "' + valid_salat + '", "precent_correct": "' + precent_correct + '", "standing":"' + count_standing + '"' \
                                                                                                                                      ',"bowing":"' + count_bowing + '","prostration" : "' + count_prostration + '","sitting":"' + count_sitting + '" }'
        result_dict = json.loads(result)
        return result_dict
    result='{"state":"error not have type"}'
    result_dict = json.loads(result)
    return result_dict

def predcition2(data="",type=""):
    result = '{"salat": "valid", "precent_correct": "100.0", "standing":"8"' \
              ',"bowing":"4","prostration" : "8","sitting":"4" }'
    result_dict = json.loads(result)

    return result_dict