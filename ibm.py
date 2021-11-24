#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from collections import Counter as c
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[5]:



import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.

if os.environ.get('RUNTIME_ENV_LOCATION_TYPE') == 'external':
    endpoint_816ffe65cd854ebea09f7ec659f25390 = 'https://s3.us.cloud-object-storage.appdomain.cloud'
else:
    endpoint_816ffe65cd854ebea09f7ec659f25390 = 'https://s3.private.us.cloud-object-storage.appdomain.cloud'

client_816ffe65cd854ebea09f7ec659f25390 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='3u5VxLqA6B-jcXF8cc5iGWEHfURMg_QCA1Ubs_8xtSYs',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url=endpoint_816ffe65cd854ebea09f7ec659f25390)

body = client_816ffe65cd854ebea09f7ec659f25390.get_object(Bucket='dynamicpricepredictionforcabs-donotdelete-pr-wrt0yfbv4mbn71',Key='cab_rides.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset = pd.read_csv(body)
dataset.head()


# In[6]:


dataset


# In[7]:


dataset.info()


# In[8]:



body = client_816ffe65cd854ebea09f7ec659f25390.get_object(Bucket='dynamicpricepredictionforcabs-donotdelete-pr-wrt0yfbv4mbn71',Key='weather.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_2 = pd.read_csv(body)
df_data_2.head()
body = client_816ffe65cd854ebea09f7ec659f25390.get_object(Bucket='dynamicpricepredictionforcabs-donotdelete-pr-wrt0yfbv4mbn71',Key='weather.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_1 = pd.read_csv(body)
df_data_1.head()
body = client_816ffe65cd854ebea09f7ec659f25390.get_object(Bucket='dynamicpricepredictionforcabs-donotdelete-pr-wrt0yfbv4mbn71',Key='weather.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset1 = pd.read_csv(body)
dataset1.head()
dataset1.info()


# In[9]:


dataset1


# # Cleaning ride data

# In[10]:


dataset


# In[11]:


dataset.isna().sum()


# In[12]:


dataset = dataset.dropna(axis=0).reset_index(drop=True)


# # Cleaning Weather data

# In[13]:


dataset1


# In[14]:


dataset1.isna().sum()


# In[15]:


dataset1=dataset1.fillna(0)


# In[16]:


#converting the timestamp data into real date format
dataset['date']=pd.to_datetime(dataset['time_stamp']/1000,unit='s')
dataset1['date']=pd.to_datetime(dataset1['time_stamp'],unit='s')


# In[17]:


# Creating the new column that contain the location and 
dataset['merged_date'] = dataset['source'].astype('str') + ' - ' + dataset['date'].dt.strftime('%Y-%m-%d').astype('str') + ' - ' + dataset['date'].dt.hour.astype('str')
dataset1['merged_date'] = dataset1['location'].astype('str') + ' - ' + dataset1['date'].dt.strftime('%Y-%m-%d').astype('str') + ' - ' + dataset1['date'].dt.hour.astype('str')


# In[18]:


#  df_rides['date'].dt.strftime('%m').head()
dataset1.index = dataset1['merged_date']


# In[19]:


# Join the weather date on rides data
df_joined = dataset.join(dataset1, on = ['merged_date'], rsuffix ='_w')


# The rides and weather data have been joined by merged_date column.

# In[20]:


df_joined.info()


# In[21]:


df_joined['id'].value_counts()


# In[22]:


df_joined[df_joined['id'] == '865b44b9-4235-4e8e-b6fd-bc8373e95b63'].iloc[:,10:22]


# In[23]:


id_group = pd.DataFrame(df_joined.groupby('id')['temp','clouds', 'pressure', 'rain', 'humidity', 'wind'].mean())
df_dataset_dataset1 = dataset.join(id_group, on = ['id'])


# In[24]:


# Creating the columns for Month, Hour and Weekdays 
df_dataset_dataset1['Month'] = df_dataset_dataset1['date'].dt.month
df_dataset_dataset1['Hour'] = df_dataset_dataset1['date'].dt.hour
df_dataset_dataset1['Day'] =  df_dataset_dataset1['date'].dt.strftime('%A')


# In[25]:


# The distribution of rides in weekdays 
import matplotlib.pyplot as plt
uber_day_count = df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Uber']['Day'].value_counts()
uber_day_count = uber_day_count.reindex(index = ['Friday','Saturday','Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday'])
lyft_day_count = df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Lyft']['Day'].value_counts()
lyft_day_count = lyft_day_count.reindex(index = ['Friday','Saturday','Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday'])

fig , ax = plt.subplots(figsize = (12,12))
ax.plot(uber_day_count.index, uber_day_count, label = 'Uber')
ax.plot(lyft_day_count.index, lyft_day_count, label = 'Lyft')
ax.set(ylabel = 'Number of Rides', xlabel = 'Weekdays')
ax.legend()
plt.show()


# In[26]:


# The ride distribution in one day 
fig , ax = plt.subplots(figsize= (12,12))
ax.plot(df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Lyft'].groupby('Hour').Hour.count().index, df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Lyft'].groupby('Hour').Hour.count(), label = 'Lyft')
ax.plot(df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Uber'].groupby('Hour').Hour.count().index, df_dataset_dataset1[df_dataset_dataset1['cab_type'] =='Uber'].groupby('Hour').Hour.count(), label = 'Uber')
ax.legend()
ax.set(xlabel = 'Hours', ylabel = 'Number of Rides')
plt.xticks(range(0,24,1))
plt.show()


# In[27]:


# The Average price of rides by type of service
import seaborn as sns

uber_order =[ 'UberPool', 'UberX', 'UberXL', 'Black','Black SUV','WAV' ]
lyft_order = ['Shared', 'Lyft', 'Lyft XL', 'Lux', 'Lux Black', 'Lux Black XL']
fig, ax = plt.subplots(2,2, figsize = (20,15))
ax1 = sns.barplot(x = df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Uber'].name, y = df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Uber'].price , ax = ax[0,0], order = uber_order)
ax2 = sns.barplot(x = df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Lyft'].name, y = df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Lyft'].price , ax = ax[0,1], order = lyft_order)
ax3 = sns.barplot(x = df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Uber'].groupby('name').name.count().index, y = df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Uber'].groupby('name').name.count(), ax = ax[1,0] ,order = uber_order)
ax4 = sns.barplot(x = df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Lyft'].groupby('name').name.count().index, y = df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Lyft'].groupby('name').name.count(), ax = ax[1,1],order = lyft_order)
for p in ax1.patches:
    ax1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
for p in ax2.patches:
    ax2.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax1.set(xlabel = 'Type of Service', ylabel = 'Average Price')
ax2.set(xlabel = 'Type of Service', ylabel = 'Average Price')
ax3.set(xlabel = 'Type of Service', ylabel = 'Number of Rides')
ax4.set(xlabel = 'Type of Service', ylabel = 'Number of Rides')
ax1.set_title('The Uber Average Prices by Type of Service')
ax2.set_title('The Lyft Average Prices by Type of Service')
ax3.set_title('The Number of Uber Rides by Type of Service')
ax4.set_title('The Number of Lyft Rides by Type of Service')
plt.show()


# In[28]:


# The average price by distance
fig , ax = plt.subplots(figsize = (12,12))
ax.plot(df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Lyft'].groupby('distance').price.mean().index, df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Lyft'].groupby('distance')['price'].mean(), label = 'Lyft')
ax.plot(df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Uber'].groupby('distance').price.mean().index, df_dataset_dataset1[df_dataset_dataset1['cab_type'] =='Uber'].groupby('distance').price.mean(), label = 'Uber')
ax.set_title('The Average Price by distance', fontsize= 15)
ax.set(xlabel = 'Distance', ylabel = 'Price' )
ax.legend()
plt.show()


# In[29]:


# The average price by distance 
fig, ax = plt.subplots(1,2 , figsize = (20,5))
for i,col in enumerate(df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Uber']['name'].unique()):
    ax[0].plot(df_dataset_dataset1[ df_dataset_dataset1['name'] == col].groupby('distance').price.mean().index, df_dataset_dataset1[ df_dataset_dataset1['name'] == col].groupby('distance').price.mean(), label = col)
ax[0].set_title('Uber Average Prices by Distance')
ax[0].set(xlabel = 'Distance in Mile', ylabel = 'Average price in USD')
ax[0].legend()
for i,col in enumerate(df_dataset_dataset1[df_dataset_dataset1['cab_type'] == 'Lyft']['name'].unique()):
    ax[1].plot(df_dataset_dataset1[ df_dataset_dataset1['name'] == col].groupby('distance').price.mean().index, df_dataset_dataset1[ df_dataset_dataset1['name'] == col].groupby('distance').price.mean(), label = col)
ax[1].set(xlabel = 'Distance in Mile', ylabel = 'Average price in USD')
ax[1].set_title('Lyft Average Prices by Distance')
ax[1].legend()
plt.show()


# In[30]:


# the average rate per mile
df_dataset_dataset1['rate_per_mile'] = round((df_dataset_dataset1['price'] / df_dataset_dataset1['distance'] ),2)
# The average rate per mile plot
fig, ax = plt.subplots(1,2,figsize = (12,5))
ax1 = sns.lineplot(x = df_dataset_dataset1.groupby(['distance'])['rate_per_mile'].mean().index, y = df_dataset_dataset1.groupby('distance')['rate_per_mile'].mean(), ax = ax[0])
ax2 = sns.lineplot(x = df_dataset_dataset1.groupby(['distance'])['rate_per_mile'].mean().index, y = df_dataset_dataset1.groupby('distance')['rate_per_mile'].mean(), ax = ax[1])
plt.xticks(range(0, 10,1))
ax1.set(xlabel = 'Distance', ylabel = 'Rate per Mile in USD')
ax2.set(xlabel = 'Distance', ylabel = 'Rate per Mile in USD', ylim = (0,15))
ax1.set_title('The Average Rate per Mile', fontsize = 16)
ax2.set_title('ZOOM Average Rate per Mile', fontsize = 16)
plt.show()


# In[31]:


# Scatter chart for Rate per mile and distance
    # pivot table to calculate average rate based on cab_type, service type(name) and distance
rates_per_mile_pivot = df_dataset_dataset1.pivot_table(index = ['cab_type', 'name', 'distance'] , values = ['rate_per_mile'])
rates_per_mile_pivot.reset_index(inplace = True)


# In[32]:


fig, ax = plt.subplots(2,2, figsize = (20,8))
ax1 = sns.scatterplot(x = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['distance'], y = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['rate_per_mile'], hue = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['name'], ax = ax[0,0])
ax2 = sns.scatterplot(x = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['distance'], y = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['rate_per_mile'], hue = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['name'], ax = ax[1,0])
ax2.set( ylim = (0,20))
ax3 = sns.scatterplot(x = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['distance'], y = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['rate_per_mile'], hue = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['name'], ax = ax[0,1])
ax4 = sns.scatterplot(x = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['distance'], y = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['rate_per_mile'], hue = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['name'], ax = ax[1,1])
ax4.set( ylim = (0,20))
handles_uber, labels_uber = ax1.get_legend_handles_labels()
handles_uber = [handles_uber[6],handles_uber[3],handles_uber[4],handles_uber[5],handles_uber[1],handles_uber[2]]
labels_uber = [labels_uber[6],labels_uber[3],labels_uber[4],labels_uber[5],labels_uber[1],labels_uber[2]]
ax1.legend(handles_uber, labels_uber)
ax2.legend(handles_uber, labels_uber)
handles_lyft, labels_lyft = ax3.get_legend_handles_labels()
handles_lyft = [handles_lyft[6],handles_lyft[4],handles_lyft[5],handles_lyft[1],handles_lyft[2],handles_lyft[3]]
labels_lyft = [labels_lyft[6],labels_lyft[4],labels_lyft[5],labels_lyft[1],labels_lyft[2],labels_lyft[3]]
ax3.legend(handles_lyft, labels_lyft)
ax4.legend(handles_lyft, labels_lyft)
ax1.set_title('Uber Rate per Mile')
ax1.set(ylabel = 'Rate per Mile in USD', xlabel = ' ')
ax2.set_title('Uber Rate Zoom(0 to 20 USD)')
ax2.set(ylabel = 'Rate per Mile in USD', xlabel = 'Distance')
ax3.set_title('Lyft Rate per Mile')
ax3.set(ylabel = ' ', xlabel = ' ')
ax4.set_title('Lyft Rate Zoom(0 to 20 USD)')
ax4.set(ylabel = ' ', xlabel = 'Distance')
plt.show()


# In[33]:


# Overrated rides
high_mile_rates = df_dataset_dataset1[df_dataset_dataset1['rate_per_mile'] > 80]
# The number of overrated rides by cab type
high_mile_rates['cab_type'].value_counts()


# In[34]:


# Overrated Lyft rides
high_mile_rates[high_mile_rates['cab_type'] == 'Lyft'].loc[:,['distance', 'cab_type', 'price', 'surge_multiplier','name', 'rate_per_mile']]


# In[35]:


# Overrated Uber Rides
high_mile_rates[high_mile_rates['cab_type'] == 'Uber'].loc[:,['distance', 'cab_type', 'price', 'surge_multiplier','name', 'rate_per_mile']].sort_values(by = 'rate_per_mile', ascending = False).head(20)


# In[36]:


# The number of rides based on service type, distance, and price 
over_rated_pivot = high_mile_rates[high_mile_rates['cab_type'] == 'Uber'].pivot_table(index = ['name', 'distance', 'price'], values = ['id'], aggfunc = len).rename(columns = {'id' : 'count_rides'})
over_rated_pivot.reset_index(inplace =True)
over_rated_pivot.sort_values(by = ['count_rides', 'name'], ascending = False).head(15)


# All of the ride distances are very short and the number of rides of one specific service type are very high. So, these are cancellations and their prices.
# 
# **Cancellation prices by service type**
# * WAV: 7.0
# * UberPool: 4.5
# * UberX: 7.0
# * UberXL: 8.5
# * Black: 15.0
# * Black SUV: 27.5
# 
# Based on these prices, if you are not ready to go, don't call Black SUV :D

# In[37]:


#before cells are testing

dataset1.groupby('location').mean()


# In[38]:


avg_dataset1 = dataset1.groupby('location').mean().reset_index(drop=False)
avg_dataset1 = avg_dataset1.drop('time_stamp', axis=1)
avg_dataset1


# # Merging Data Frames

# In[39]:


dataset = dataset.drop('merged_date', axis=1)
dataset = dataset.drop('date', axis=1)
dataset


# In[40]:


dataset1 = dataset1.drop('merged_date', axis=1)
dataset1 = dataset1.drop('date', axis=1)
dataset1


# In[41]:


source_dataset1 = avg_dataset1.rename(
    columns={
        'location': 'source',
        'temp': 'source_temp',
        'clouds': 'source_clouds',
        'pressure': 'source_pressure',
        'rain': 'source_rain',
        'humidity': 'source_humidity',
        'wind': 'source_wind'
    }
)

source_dataset1


# In[42]:


destination_dataset1 = avg_dataset1.rename(
    columns={
        'location': 'destination',
        'temp': 'destination_temp',
        'clouds': 'destination_clouds',
        'pressure': 'destination_pressure',
        'rain': 'destination_rain',
        'humidity': 'destination_humidity',
        'wind': 'destination_wind'
    }
)

destination_dataset1


# In[43]:


data = dataset    .merge(source_dataset1, on='source')    .merge(destination_dataset1, on='destination')

data


# In[44]:


data.name.unique()


# In[45]:


data.source.unique()


# In[46]:


item_counts = data["source"].value_counts()
item_counts


# In[47]:


data.destination.unique()


# In[48]:


item_counts = data["destination"].value_counts()
item_counts


# In[49]:


data.product_id.unique()


# In[50]:


item_counts = data["name"].value_counts()
item_counts


# In[51]:


item_counts = data["product_id"].value_counts()
item_counts


# In[52]:


cat=data.dtypes[data.dtypes=='O'].index.values
cat


# In[53]:


from collections import Counter as c # return counts
for i in cat:
    print("Column :",i)
    print('count of classes : ',data[i].nunique())
    print(c(data[i]))
    print('*'*120)


# In[54]:


data.dtypes[data.dtypes!='O'].index.values


# In[55]:


data.isnull().any()#it will return true if any columns is having null values


# In[56]:


data.isnull().sum() #used for finding the null values


# # Label Encoding

# In[57]:


data1=data.copy()
from sklearn.preprocessing import LabelEncoder #importing the LabelEncoding from sklearn
x='*'
for i in cat:#looping through all the categorical columns
    print("LABEL ENCODING OF:",i)
    LE = LabelEncoder()#creating an object of LabelEncoder
    print(c(data[i])) #getting the classes values before transformation
    data[i] = LE.fit_transform(data[i]) # trannsforming our text classes to numerical values
    print(c(data[i])) #getting the classes values after transformation
    print(x*100)


# In[58]:


data.head()


# In[59]:


data.info()


# In[60]:


x = data.drop(['price','distance','time_stamp','surge_multiplier','id','source_temp','source_clouds','source_pressure','source_rain','source_humidity','source_wind','destination_temp','destination_clouds','destination_pressure','destination_rain','destination_humidity','destination_wind'],axis=1) #independet features
x=pd.DataFrame(x)
y = data['price'] #dependent feature
y=pd.DataFrame(y)


# In[61]:


x.head()


# In[62]:


y.head()


# # Splitting dataset into train and test

# In[63]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train.shape)
print(x_test.shape)


# In[64]:


from sklearn.ensemble import RandomForestRegressor
rand=RandomForestRegressor(n_estimators=20,random_state=52,n_jobs=-1,max_depth=4)
rand.fit(x_train,y_train)


# In[80]:


x_train.shape


# # Predicting the result

# In[66]:


ypred=rand.predict(x_test)
print(ypred)


# # Score of the model

# In[67]:


rand.score(x_train,y_train)


# # Saving our model

# In[68]:


import pickle
pickle.dump(rand, open("model.pkl", "wb"))


# In[69]:


get_ipython().system('pip install ibm_watson_machine_learning')


# In[70]:


from ibm_watson_machine_learning import APIClient
wml_credentials = {
                   "url": "https://us-south.ml.cloud.ibm.com",
                   "apikey":"l0ArGgOyZ3AWyBtW5SQG-9PPK8SgWoamljoLcqyw3CaR"
                  }
client = APIClient(wml_credentials)


# In[71]:


def guid_from_space_name(client, space_name):
    space = client.spaces.get_details()
    #print(space)
    return(next(item for item in space['resources'] if item['entity']["name"] == space_name)['metadata']['id'])


# In[72]:


space_uid = guid_from_space_name(client,'models')
print("Space UID=" + space_uid)


# In[73]:


client.set.default_space(space_uid)


# In[74]:


client.software_specifications.list()


# In[75]:


software_spec_uid = client.software_specifications.get_uid_by_name("default_py3.8")
software_spec_uid


# In[76]:


model_details = client.repository.store_model(model=rand,meta_props={
client.repository.ModelMetaNames.NAME:"cab_rides",
client.repository.ModelMetaNames.TYPE:"scikit-learn_0.23",
client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:software_spec_uid }

                                             )     
model_id =client.repository.get_model_uid(model_details)                                              


# In[77]:


model_id


# In[ ]:




