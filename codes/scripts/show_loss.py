from matplotlib import rcParams
import matplotlib.pyplot as plt
import re
##显示中文
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'SimSun,Times New Roman'

##读取log文件
logFile = r'../experiments/PDM-SR/myDataset_deg_new_v3_mse100/log/train_myDataset_deg_new_v3_mse100_rank0_230406-144702.log' 
text = ''
file = open(logFile)
for line in file:
    text += line
file.close()
all_list = re.findall('g1_adv: .*[0-9]',text)  


train_g1loss = []
k=0
# for i in all_list:
#     if(k<100):
#         train_g1loss.append(float(i.split('g1_adv: ')[1].split('; d1_adv: ')[0]))
#         k+=1
# train_d1loss = []
# k=0
# for i in all_list:
#     if k<100:
#         train_d1loss.append(float(i.split('; d1_adv: ')[1]))
#         k+=1
train_g1loss = []
for i in all_list:
    train_g1loss.append(float(i.split('g1_adv: ')[1].split('; noise_mean: ')[0]))

train_noiseloss = []
for i in all_list:
    train_g1loss.append(float(i.split('; noise_mean: ')[1].split('; d1_adv:')[0]))

train_d1loss = []
for i in all_list:
    train_d1loss.append(float(i.split('; d1_adv: ')[1]))



plt.plot(train_g1loss,label='train_g1loss')
plt.plot(train_noiseloss,label='train_noiseloss')

plt.plot(train_d1loss,label='train_d1loss')
# plt.plot(train_acc,label='train_acc')
# plt.plot(val_acc,label='val_acc')
plt.legend(loc='best')
plt.show()
