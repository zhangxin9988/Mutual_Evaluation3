import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
data_origin=pd.read_csv("C:/ѧϰ/�칫/�о����γ�/��һ�ڶ�ѧ�ڿγ�/�����ھ�/hotel_bookings.csv")
print(data_origin.head(10))
#�����ݼ�һ����32�У�����û�ж����е��н��з���������ѡ��һ���֡�
data=pd.DataFrame(data_origin[['hotel','is_canceled','arrival_date_year','arrival_date_month','arrival_date_day_of_month','meal','previous_bookings_not_canceled','reserved_room_type','customer_type','adr','reservation_status','reservation_status_date']])
print(data.head(10))
data.replace('Undefined','SC',inplace=True) #��mealһ��ֵΪUndefinedȫ���滻ΪSC
#�������Ƶ�����ݷֿ����棬��������ıȽ�
data_rh=data[(data.hotel=='Resort Hotel') ]
data_ch=data[(data.hotel=='City Hotel') ]
#Ԥ���������ͱȽ�
plt.figure(figsize=(20, 8))
room_type=set(list(data_rh.reserved_room_type.value_counts().index)+list(data_ch.reserved_room_type.value_counts().index))
room_type=sorted(list(room_type))
print(room_type)
x=np.arange(len(room_type))
bar_width=0.3
plt.bar(x,data_rh.reserved_room_type.value_counts()[room_type],bar_width,color='r', label='Resort Hotel')
plt.bar(x+bar_width,data_ch.reserved_room_type.value_counts()[room_type],bar_width,color='y', label='City Hotel')
plt.xlabel('reserved_room_type')
plt.ylabel('numbers')
plt.legend(['Resort Hotel','City Hotel'])
plt.xticks(x+bar_width/2,room_type)
plt.show()

#��ס�ʱȽ�
#��ס��=Ԥ��δȡ����¼��/�ܼ�¼��
occupancy_rh=data_rh[data_rh.is_canceled==0].is_canceled.count()/data_rh.is_canceled.count()
occupancy_ch=data_ch[data_ch.is_canceled==0].is_canceled.count()/data_ch.is_canceled.count()
print('���վƵ����ס��={:.3}'.format(occupancy_rh))
print('���оƵ����ס��={:.3}'.format(occupancy_ch))

#��ǰԤ��ʱ�����
data_lead=data_origin[['hotel','is_canceled','lead_time'  ]]
data_lead_rh=data_lead[(data_lead.hotel=='Resort Hotel')& data_lead.is_canceled==0]
data_lead_ch=data_lead[(data_lead.hotel=='City Hotel')& data_lead.is_canceled==0]
#���Ƽ��վƵ�ͳ��оƵ���ǰԤ��ʱ�����Ӧ��¼������ɢ��ͼ
plt.figure(figsize=(18, 8))
rh_lead_time=data_lead_rh.lead_time.value_counts().sort_index()
ch_lead_time=data_lead_ch.lead_time.value_counts().sort_index()
plt.plot(rh_lead_time[0:20].index,rh_lead_time[0:20],marker='o',label='Resort Hotel')
plt.plot(ch_lead_time[0:20].index,ch_lead_time[0:20],marker='v',label='City Hotel')
plt.legend()
plt.xlabel('lead_time')
plt.ylabel('counts')
plt.show()

#��סʱ������ ��סʱ��=Ԥ�����״̬ʱ��-����ʱ��
data_bookdate=pd.DataFrame(data[['arrival_date_year','arrival_date_month','arrival_date_day_of_month','reservation_status_date']])
data_bookdate.head(10)

#���·ݱ�ʾΪ�ַ���
class_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9,'October': 10,'November': 11,'December': 12}
data_bookdate.arrival_date_month=data_bookdate.arrival_date_month.map(class_mapping)
data_bookdate['reservation_status_date']=pd.to_datetime(data_bookdate['reservation_status_date'])#ת��Ϊ���ڸ�ʽ
data_bookdate['reservation_status_date_year']=data_bookdate['reservation_status_date'].dt.year
data_bookdate['reservation_status_date_month']=data_bookdate['reservation_status_date'].dt.month
data_bookdate['reservation_status_date_day']=data_bookdate['reservation_status_date'].dt.day
data_bookdate.drop(columns=['reservation_status_date'],inplace=True)
data_bookdate.head(10)

import datetime
book_days_list=[]
for i in range(0, len(data_bookdate)):
    a_date=datetime.date(data_bookdate.iloc[i]['arrival_date_year'],data_bookdate.iloc[i]['arrival_date_month'],data_bookdate.iloc[i]['arrival_date_day_of_month'])
    r_date=datetime.date(data_bookdate.iloc[i]['reservation_status_date_year'],data_bookdate.iloc[i]['reservation_status_date_month'],data_bookdate.iloc[i]['reservation_status_date_day'])
    book_days_list.append((r_date-a_date).days) #����ÿ����¼����סʱ��
book_days=pd.Series(book_days_list)
print(book_days.value_counts()[0:20])

#������סʱ�����Ӧ��¼����������ͳ��ͼ
plt.figure(figsize=(15, 8))
plt.bar(book_days.value_counts()[0:20].index,book_days.value_counts()[0:20],width=0.5,color='red')
plt.xlabel('reservation_date - arrive_date')
plt.ylabel('counts')
plt.show()

#Ԥ���������
data_repeat=pd.DataFrame( data_origin[['is_repeated_guest','agent','reservation_status_date']] )
#��agent ID=240�Ŀͻ�Ϊ����
data_repeat=pd.DataFrame(data_repeat[(data_repeat.is_repeated_guest==1)&(data_repeat.agent==240) ] )
data_repeat.drop_duplicates('reservation_status_date',inplace=True) #ɾ��������ֵ��ͬ�ļ�¼
data_repeat['reservation_status_date']=pd.to_datetime(data_repeat['reservation_status_date'])#ת��Ϊ���ڸ�ʽ
data_repeat['year']=data_repeat['reservation_status_date'].dt.year
data_repeat['month']=data_repeat['reservation_status_date'].dt.month
data_repeat['day']=data_repeat['reservation_status_date'].dt.day
data_repeat.drop(columns=['reservation_status_date'],inplace=True)
book_repeat_list=[]
for i in range(0, len(data_repeat)):
    repeat=datetime.date(int(data_repeat.iloc[i]['year']),int(data_repeat.iloc[i]['month']),int(data_repeat.iloc[i]['day']) )
    book_repeat_list.append(repeat)
book_repeat_list.sort()
gap={}
for index,i in enumerate(book_repeat_list):
    if index==len(book_repeat_list)-1:
        break
    gap[index]= (book_repeat_list[index+1]-book_repeat_list[index]).days
#����ÿ��Ԥ���������������ͼ
plt.figure(figsize=(15, 8))
plt.plot(list(gap.keys()),list(gap.values()),'g.-')
plt.xlabel('number of times')
plt.ylabel('gap numbers')
plt.show()

#�������ͱȽ�
bar_width = 0.3 # ���ο��
index_rh = np.arange(len(data_rh.meal.value_counts().index))
index_ch = index_rh + bar_width
plt.figure(figsize=(15, 8))
plt.bar(index_rh,data_rh.meal.value_counts(),width=bar_width,color='gray', label='Resort Hotel')
plt.bar(index_ch,data_ch.meal.value_counts(),width=bar_width,color='pink', label='City Hotel')
plt.legend()
plt.xticks(index_rh + bar_width/2, data_rh.meal.value_counts().index)  # �ú�������̶���ʾdata_rh.meal.value_counts().index�� index_rh + bar_width/2 Ϊ��������̶ȵ�λ��
plt.xlabel('meal')
plt.ylabel('numbers')
plt.title('comparison of meal')
data_bestBooking=pd.DataFrame(data_origin[['hotel','is_canceled','arrival_date_year','arrival_date_month','arrival_date_day_of_month']])
#���ȿ����վƵ�ͳ��оƵ�����Ԥ���·�
data_bestBooking_rh=data_bestBooking[(data_bestBooking.hotel=='Resort Hotel')&(data_bestBooking.is_canceled==0)]
data_bestBooking_ch=data_bestBooking[(data_bestBooking.hotel=='City Hotel')&(data_bestBooking.is_canceled==0)]
plt.figure(figsize=(20, 8))
rh_arrival_month=data_bestBooking_rh.arrival_date_month.value_counts()
rh_arrival_month=pd.Series(rh_arrival_month,index=['January','February','March','April','May','June','July','August','September','October','November','December'])
ch_arrival_month=data_bestBooking_ch.arrival_date_month.value_counts()
ch_arrival_month=pd.Series(ch_arrival_month,index=['January','February','March','April','May','June','July','August','September','October','November','December'])
bar_width=0.4
x=np.arange(12)
plt.bar(x,rh_arrival_month,bar_width,color='pink', label='Resort Hotel')
plt.bar(x+bar_width,ch_arrival_month,bar_width,color='gray', label='City Hotel')
plt.xlabel('booking month')
plt.ylabel('numbers')
plt.xticks(x+bar_width/2,ch_arrival_month.index)
plt.legend(['Resort Hotel','City Hotel'])
plt.title('Resort Hotel V.S. City Hotel')
data_bestBooking_rh_day=data_bestBooking[(data_bestBooking.hotel=='Resort Hotel')&(data_bestBooking.is_canceled==0)&(data_bestBooking.arrival_date_month=='August')]
data_bestBooking_ch_day=data_bestBooking[(data_bestBooking.hotel=='City Hotel')&(data_bestBooking.is_canceled==0)&(data_bestBooking.arrival_date_month=='August')]
plt.figure(figsize=(20, 8))
plt.subplot(1,2,1)
plt.bar(data_bestBooking_rh_day.arrival_date_day_of_month.value_counts()[0:10].index,data_bestBooking_rh.arrival_date_day_of_month.value_counts()[0:10],width=0.6,color='pink', label='Resort Hotel')
plt.xlabel('booking day')
plt.ylabel('numbers')
plt.title('Resort Hotel')
plt.subplot(1,2,2)
plt.bar(data_bestBooking_ch_day.arrival_date_day_of_month.value_counts()[0:10].index,data_bestBooking_ch.arrival_date_day_of_month.value_counts()[0:10],width=0.6,color='gray', label='City Hotel')
plt.xlabel('booking day')
plt.ylabel('numbers')
plt.title('City Hotel')
### ����������ϢԤ��Ƶ�Ԥ�������״̬
# 0-�ͻ�����ס���˷�
# # 1--Ԥ��ȡ��
# # 2--����δ������ס����
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data_LR=pd.DataFrame(data_origin[['hotel','is_canceled','lead_time','adults','children','babies','meal','previous_cancellations','previous_bookings_not_canceled','booking_changes','adr','reservation_status']])
data_LR.replace('Undefined','SC',inplace=True) #��mealһ��ֵΪUndefinedȫ���滻ΪSC
class_mapping1 = {'Resort Hotel': 0,'City Hotel':1}
class_mapping2 = {'Check-Out':0,'Canceled':1,'No-Show':2}
data_LR.hotel=data_LR.hotel.map(class_mapping1)
data_LR.reservation_status=data_LR.reservation_status.map(class_mapping2)
meal_onehot=pd.get_dummies(data_LR.meal)
data_bookdate.head(10)
data_LR=pd.concat([data_LR,meal_onehot,data_bookdate],axis = 1)
data_LR.drop(['meal'], axis=1,inplace=True)
data_LR.adr=data_LR.adr.astype('int')
data_LR.dropna(inplace=True)
data_LR.head(10)
y = data_LR['reservation_status']  #��reservation_statusһ�е�ֵ����ǩ
x = data_LR.drop('reservation_status',axis=1)
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)
logreg=LogisticRegression().fit(X_train,y_train)
plt.figure(figsize=(12, 8))
plt.plot(logreg.coef_.T,'v')
plt.xticks(range(x.shape[1]),x.columns,rotation=90)
plt.ylim(-5,5)
plt.xlabel('columns names')
plt.ylabel('para values')
plt.show()
