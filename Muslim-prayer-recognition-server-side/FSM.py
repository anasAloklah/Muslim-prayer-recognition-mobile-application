from collections import Counter
def FSM_asar(list_activties=[]):
    if(FSM_Rokaa(list_activties[0:6])!=True): #first Rokaa
        return False
    if (FSM_Rokaa(list_activties[6:12])!=True): #secund Rokaa
        return False
    if (list_activties[12] != 2):  # Sitting
        return False
    if (FSM_Rokaa(list_activties[13:19])!=True): #third Rokaa
        return False
    if (FSM_Rokaa(list_activties[19:25])!=True): #forth Rokaa
        return False
    if (list_activties[25] != 2):  # Sitting
        return False
    return True
def FSM_fajer(list_activties=[]):
    if (FSM_Rokaa(list_activties[0:6]) != True):  # first Rokaa
        return False
    if (FSM_Rokaa(list_activties[6:12]) != True):  # secund Rokaa
        return False
    if (list_activties[12] != 2):  # Sitting
        return False
    return True

def FSM_magrab(list_activties=[]):
    if (FSM_Rokaa(list_activties[0:6]) != True):  # first Rokaa
        return False
    if (FSM_Rokaa(list_activties[6:12]) != True):  # secund Rokaa
        return False
    if (list_activties[12] != 2):  # Sitting
        return False
    if (FSM_Rokaa(list_activties[13:19]) != True):  # third Rokaa
        return False
    if (list_activties[19] != 2):  # Sitting
        return False
    return True

def FSM_Rokaa(list_activties=[]):

    count=0
    if(list_activties[count]!=3 or list_activties[count] != 3):#standing
        return False
    count=count+1
    if (list_activties[count] != 0):# Bowing
        return False
    count = count + 1
    if (list_activties[count] != 3):  # standing
        return False
    count = count + 1
    if (list_activties[count] != 1):  # prostration
        return False
    count = count + 1
    if (list_activties[count] != 2):  # Sitting
        return False
    count = count + 1
    if (list_activties[count] != 1):  # prostration
        return False
    return True

def similarity2(list1=[],list2=[]):
    res = len(set(list1) & set(list2)) / float(len(set(list1) | set(list2))) * 100
    return res
def similarity(list1=[],list2=[]):
    res = ((len(list1)-diffrantcy2(list1,list2))/len(list1)) * 100
    return res

def diffrantcy(list1=[],list2=[]):
    diff=list(Counter(list1) - Counter(list2))
    return len(diff)
def diffrantcy2(O_list1=[],P_list2=[]):
    count=0
    len_1=min(len(O_list1),len(P_list2))
    for i in range(0, len_1):
        if(O_list1[i]!=P_list2[i]):
            count=count+1
    if (len(O_list1)!=len(P_list2)):
        count=count+abs(len(O_list1)-len(P_list2))
    return count
def remove_redance(list_activties=[]):
    new_list=[]
    j=0
    for i in range(0,len(list_activties)):
        if(i==0):
            new_list.append(list_activties[0])
        if  (list_activties[i]!=new_list[j]):
            new_list.append(list_activties[i])
            j=j+1
    return new_list
def simlarty_asar(list_activties=[]):
    list_of_asar = [3, 0, 3, 1, 2, 1, 3, 0, 3, 1, 2, 1, 2, 3, 0, 3, 1, 2, 1, 3, 0, 3, 1, 2, 1, 2]
    sim=similarity(list_of_asar, list_activties)
    return sim
def simlarty_fajer(list_activties=[]):
    list_of_fajer = [3, 0, 3, 1, 2, 1, 3, 0, 3, 1, 2, 1, 2]
    sim=similarity(list_of_fajer, list_activties)
    return sim
def simlarty_magrab(list_activties=[]):
    list_of_magram = [3, 0, 3, 1, 2, 1, 3, 0, 3, 1, 2, 1, 2, 3, 0, 3, 1, 2, 1, 2]
    sim=similarity(list_of_magram, list_activties)
    return sim
def remove_noise(list_activties=[],frame_noise=3):
    for i in range(0, len(list_activties)-frame_noise):
        if(len(remove_redance(list_activties[i:i+frame_noise]))>2):
            for j in range(i,i+frame_noise):
                list_activties[j]=list_activties[i]
    return list_activties
def corrctor (list_activties=[]):
    ind=0
    for i in range(0, len(list_activties)):
        if( list_activties[i]==3):
            ind=i
            break
    list_activties=list_activties[ind:len(list_activties)]

    for i in range(0, len(list_activties)-2):
        if( (list_activties[i]==3 and list_activties[i+2]==3)):
            if(list_activties[i+1]!=0 and list_activties[i+1]!=1 ):
                list_activties[i+1]=0
        if ((list_activties[i] == 1 and list_activties[i + 2] == 1)):
            if (list_activties[i+1] != 2):
                list_activties[i+1]=2
    return list_activties