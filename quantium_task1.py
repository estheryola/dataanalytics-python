#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Data analysis and wrangling
import pandas as pd
import numpy as np

# Data visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# Dates
import datetime
from matplotlib.dates import DateFormatter

# Text analysis
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist as fdist
import re

# Statistical analysis
from scipy.stats import ttest_ind

# Remove warnings
import warnings
warnings.filterwarnings('ignore')


# In[17]:


pwd


# In[23]:


customerdata = pd.read_csv("D:/New Levels/Quantium/QVI_purchase_behaviour.csv")
transactiondata = pd.read_csv("D:/New Levels/Quantium/QVI_transaction_data.csv")


# In[26]:


transactiondata.head()


# In[30]:


len(transactiondata)


# In[31]:


transactiondata('TXN_ID')


# In[32]:


transactiondata('TXN_ID').numunique()


# In[33]:


transactiondata('TXN_ID').nunique()


# In[34]:


transactionData['TXN_ID'].nunique()


# In[35]:


transactiondata['TXN_ID'].nunique()


# In[37]:


transactiondata[transactiondata.duplicated(['TXN_ID'])].head()


# In[39]:


transactiondata.loc[transactiondata['TXN_ID'] == 48887, :]


# In[40]:


transactiondata.info()


# In[42]:


list(transactiondata.columns)


# In[45]:


transactiondata['DATE'].head()


# In[46]:


def xlseriesdate_to_datetime(xlserialdate):
    excel_anchor = datetime.datetime(1900,1,1)
    if(xlserialdate < 60):
        delta_in_days = datetime.timedelta(days = (xlserialdate - 1))
    else:
        delta_in_days = datetime.timedelta(days = (xlserialdate - 2))
        converted_date = excel_anchor + delta_in_days
    return converted_date


# In[47]:


transactiondata['DATE'] = transactiondata['DATE'].apply(xlseriesdate_to_datetime)


# In[49]:


transactiondata['DATE'].head()


# In[50]:


transactiondata.head()


# In[51]:


transactiondata['PROD_NAME'].head()


# In[52]:


transactiondata['PROD_SIZE'] = transactiondata['PROD_NAME'].str.extract("(\d+)")
transactiondata['PROD_SIZE'] = pd.to_numeric(transactiondata['PROD_SIZE'])


# In[53]:


transactiondata.head()


# In[57]:


def clean_text(text):
    text = re.sub('[&/]', ' ', text)
    text = re.sub('\d\w*', ' ', text)
    return text


# In[58]:


transactiondata['PROD_NAME'] = transactiondata['PROD_NAME'].apply(clean_text)


# In[59]:


transactiondata['PROD_NAME'].head()


# In[60]:


transactiondata.head()


# In[65]:


cleanprodname = transactiondata['PROD_NAME']
string = ''.join(cleanprodname)
prodword = word_tokenize(string)


# In[66]:


wordfreq = fdist(prodword)
freq_df = pd.DataFrame(list(wordfreq.items()), columns = ["Word", "Frequency"]).sort_values(by = 'Frequency', ascending = False)


# In[67]:


wordfreq = fdist(prodword)
freq_df = pd.DataFrame(list(wordfreq.items()), columns = ["Word", "Frequency"]).sort_values(by = 'Frequency', ascending = False)


# In[68]:


freq_df.head()


# In[69]:


transactiondata['PROD_NAME'] = transactiondata['PROD_NAME'].apply(lambda x: x.lower())
transactiondata = transactiondata[~transactiondata['PROD_NAME'].str.contains("salsa")]
transactiondata['PROD_NAME'] = transactiondata['PROD_NAME'].apply(lambda x: x.title())


# In[70]:


transactiondata.head()


# In[72]:


transactiondata['PROD_QTY'].value_counts()


# In[73]:


transactiondata.loc[transactiondata['PROD_QTY'] == 200, :]


# In[75]:


transactiondata.drop(transactiondata.index[transactiondata['LYLTY_CARD_NBR'] == 226000], inplace = True)
customerdata.drop(customerdata.index[customerdata['LYLTY_CARD_NBR'] == 226000], inplace = True)


# In[78]:


transactiondata.loc[transactiondata['LYLTY_CARD_NBR'] == 226000, :]


# In[79]:


transactiondata['DATE'].nunique()


# In[80]:


pd.date_range(start = '2018-07-01', end = '2019-06-30').difference(transactiondata['DATE'])


# In[81]:


a = pd.pivot_table(transactiondata, values = 'TOT_SALES', index = 'DATE', aggfunc = 'sum')
a.head()


# In[84]:


b = pd.DataFrame(index = pd.date_range(start = '2018-07-01', end = '2019-06-30'))
b['TOT_SALES'] = 0
len(b)


# In[85]:


c = a + b
c.fillna(0, inplace = True)


# In[86]:


c.head()


# In[89]:


c.index.name = 'Date'
c.rename(columns = {'TOT_SALES': 'Total Sales'}, inplace = True)
c.head()


# In[92]:


timeline = c.index
graph = c['Total Sales']

fig, ax = plt.subplots(figsize = (10, 5))
ax.plot(timeline, graph)

date_form = DateFormatter("%Y-%m")
ax.xaxis.set_major_formatter(date_form)
plt.title('Total Sales from July 2018 to June 2019')
plt.xlabel('Time')
plt.ylabel('Total Sales')

plt.show()


# In[93]:


c[c['Total Sales'] == 0]


# In[94]:


c_december = c[(c.index < "2019-01-01") & (c.index > "2018-11-30")]
c_december.head()


# In[95]:


plt.figure(figsize = (15, 5))
plt.plot(c_december)
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('Total Sales in December')


# In[96]:


c_december.reset_index(drop = True, inplace = True)
c_december.head()


# In[97]:


c_december['Date'] = c_december.index + 1
c_december.head()


# In[98]:


plt.figure(figsize = (15,5))
sns.barplot(x = 'Date', y ='Total Sales', data = c_december)


# In[99]:


transactiondata['PROD_SIZE'].head()


# In[100]:


transactiondata['PROD_SIZE'].unique()


# In[101]:


transactiondata['PROD_SIZE'].hist()


# In[103]:


part = transactiondata['PROD_NAME'].str.partition()
transactiondata['BRAND'] = part[0]
transactiondata.head()


# In[104]:


transactiondata['BRAND'].unique()


# In[106]:


transactiondata['BRAND'].replace('Ncc', 'Natural', inplace = True)
transactiondata['BRAND'].replace('Ccs', 'CCS', inplace = True)
transactiondata['BRAND'].replace('Smith', 'Smiths', inplace = True)
transactiondata['BRAND'].replace(['Grain', 'Grnwves'], 'Grainwaves', inplace = True)
transactiondata['BRAND'].replace('Dorito', 'Doritos', inplace = True)
transactiondata['BRAND'].replace('Ww', 'Woolworths', inplace = True)
transactiondata['BRAND'].replace('Infzns', 'Infuzions', inplace = True)
transactiondata['BRAND'].replace(['Red', 'Rrd'], 'Red Rock Deli', inplace = True)
transactiondata['BRAND'].replace('Snbts', 'Sunbites', inplace = True)

transactiondata['BRAND'].unique()


# In[107]:


transactiondata.groupby('BRAND').TOT_SALES.sum().sort_values(ascending = False)


# In[108]:


list(customerdata.columns)


# In[109]:


customerdata.head()


# In[110]:


len(customerdata)


# In[111]:


customerdata['LYLTY_CARD_NBR'].nunique()


# In[112]:


customerdata['LIFESTAGE'].nunique()


# In[113]:


customerdata['LIFESTAGE'].unique()


# In[114]:


customerdata['LIFESTAGE'].value_counts().sort_values(ascending = False)


# In[115]:


sns.countplot(y = customerdata['LIFESTAGE'], order = customerdata['LIFESTAGE'].value_counts().index)


# In[116]:


customerdata['PREMIUM_CUSTOMER'].nunique()


# In[117]:


customerdata['PREMIUM_CUSTOMER'].unique()


# In[119]:


customerdata['PREMIUM_CUSTOMER'].value_counts().sort_values(ascending = False)


# In[121]:


plt.figure(figsize = (12, 7))
sns.countplot(y = customerdata['PREMIUM_CUSTOMER'], order = customerdata['PREMIUM_CUSTOMER'].value_counts().index)
plt.xlabel('Number of Customers')
plt.ylabel('Premium Customer')


# In[122]:


transactiondata.shape


# In[123]:


customerdata.shape


# In[124]:


combinedata = pd.merge(transactiondata, customerdata)
combinedata.shape


# In[125]:


combinedata.head()


# In[130]:


sales = pd.DataFrame(combinedata.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).TOT_SALES.sum())
sales.rename(columns = {'TOT_SALES': 'Total Sales'}, inplace = True)
sales.sort_values(by = 'Total Sales', ascending = False, inplace = True)
sales


# In[131]:


salesPlot = pd.DataFrame(combinedata.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).TOT_SALES.sum())
salesPlot.unstack().plot(kind = 'bar', stacked = True, figsize = (12, 7), title = 'Total Sales by Customer Segment')
plt.ylabel('Total Sales')
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 2)


# In[132]:


customers = pd.DataFrame(combinedata.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).LYLTY_CARD_NBR.nunique())
customers.rename(columns = {'LYLTY_CARD_NBR': 'Number of Customers'}, inplace = True)
customers.sort_values(by = 'Number of Customers', ascending = False).head(10)


# In[133]:


customersPlot = pd.DataFrame(combinedata.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).LYLTY_CARD_NBR.nunique())
customersPlot.unstack().plot(kind = 'bar', stacked = True, figsize = (12, 7), title = 'Number of Customers by Customer Segment')
plt.ylabel('Number of Customers')
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 2)


# In[135]:


avg_units = combinedata.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).PROD_QTY.sum() / combinedata.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).LYLTY_CARD_NBR.nunique()
avg_units = pd.DataFrame(avg_units, columns = {'Average Unit per Customer'})
avg_units.sort_values(by = 'Average Unit per Customer', ascending = False).head()


# In[136]:


avgUnitsPlot = pd.DataFrame(combinedata.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).PROD_QTY.sum() / combinedata.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).LYLTY_CARD_NBR.nunique())
avgUnitsPlot.unstack().plot(kind = 'bar', figsize = (12, 7), title = 'Average Unit by Customer Segment')
plt.ylabel('Average Number of Units')
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 2)


# In[137]:


avg_price = combinedata.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).TOT_SALES.sum() / combinedata.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).PROD_QTY.sum()
avg_price = pd.DataFrame(avg_price, columns = {'Price per Unit'})
avg_price.sort_values(by = 'Price per Unit', ascending = False).head()


# In[138]:


avgPricePlot = pd.DataFrame(combinedata.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).TOT_SALES.sum() / combinedata.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).PROD_QTY.sum())
avgPricePlot.unstack().plot(kind = 'bar', figsize = (12, 7), title = 'Average Price by Customer Segment', ylim = (0, 6))
plt.ylabel('Average Price')
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 2)


# In[139]:


pricePerUnit = combinedata
pricePerUnit['PRICE'] = pricePerUnit['TOT_SALES'] / pricePerUnit['PROD_QTY']
pricePerUnit.head()


# In[140]:


mainstream = pricePerUnit.loc[(pricePerUnit['PREMIUM_CUSTOMER'] == 'Mainstream') & ( (pricePerUnit['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') | (pricePerUnit['LIFESTAGE'] == 'MIDAGE SINGLES/COUPLES') ), 'PRICE']
nonMainstream = pricePerUnit.loc[(pricePerUnit['PREMIUM_CUSTOMER'] != 'Mainstream') & ( (pricePerUnit['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') | (pricePerUnit['LIFESTAGE'] == 'MIDAGE SINGLES/COUPLES') ), 'PRICE']


# In[141]:


plt.figure(figsize = (10, 5))
plt.hist(mainstream, label = 'Mainstream')
plt.hist(nonMainstream, label = 'Premium & Budget')
plt.legend()
plt.xlabel('Price per Unit')


# In[142]:


print("Mainstream average price per unit: ${:.2f}".format(np.mean(mainstream)))
print("Non-mainstream average price per unit: ${:.2f}".format(np.mean(nonMainstream)))
if np.mean(mainstream) > np.mean(nonMainstream):
    print("Mainstream customers have higher average price per unit. ")
else:
    print("Non-mainstream customers have a higher average price per unit. ")


# In[143]:


ttest_ind(mainstream, nonMainstream)


# In[145]:


target = combinedata.loc[(combinedata['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') & (combinedata['PREMIUM_CUSTOMER'] == 'Mainstream'), :]
nonTarget = combinedata.loc[(combinedata['LIFESTAGE'] != 'YOUNG SINGLES/COUPLES' ) & (combinedata['PREMIUM_CUSTOMER'] != 'Mainstream'), :]
target.head()


# In[146]:


targetBrand = target.loc[:, ['BRAND', 'PROD_QTY']]
targetSum = targetBrand['PROD_QTY'].sum()
targetBrand['Target Brand Affinity'] = targetBrand['PROD_QTY'] / targetSum
targetBrand = pd.DataFrame(targetBrand.groupby('BRAND')['Target Brand Affinity'].sum())


# In[147]:


nonTargetBrand = nonTarget.loc[:, ['BRAND', 'PROD_QTY']]
nonTargetSum = nonTargetBrand['PROD_QTY'].sum()
nonTargetBrand['Non-Target Brand Affinity'] = nonTargetBrand['PROD_QTY'] / nonTargetSum
nonTargetBrand = pd.DataFrame(nonTargetBrand.groupby('BRAND')['Non-Target Brand Affinity'].sum())


# In[148]:


brand_proportions = pd.merge(targetBrand, nonTargetBrand, left_index = True, right_index = True)
brand_proportions.head()


# In[149]:


brand_proportions['Affinity to Brand'] = brand_proportions['Target Brand Affinity'] / brand_proportions['Non-Target Brand Affinity']
brand_proportions.sort_values(by = 'Affinity to Brand', ascending = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




