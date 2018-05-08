
# coding: utf-8

# In[ ]:


# I'm just taking these three results and naively mixing them together through different kinds of means
import pandas as pd
import numpy as np


# In[ ]:


print("Reading the data...\n")
df1 = pd.read_csv('../output/input_ensemble/_w9769_wordbatch_9769.csv')
df2 = pd.read_csv('../output/input_ensemble/_w9795_20180419_gbdt_removeday_sub_it_50percent_10_180000010_31_6_9795.csv')
df3 = pd.read_csv('../output/input_ensemble/_w9795_gbdt_sub_it_50percent_10_180000010_31_6_9795.csv')
df4 = pd.read_csv('../output/input_ensemble/_w9795_gdbt_sub_it_50percent_10_180000010_31_5_9795.csv')

# In[ ]:


models = {  'df1' : {
                    'name':'wordbatch',
                    'score':97.69,
                    'df':df1 },
            'df2' : {
                    'name':'light1',
                    'score':97.95,
                    'df':df2 },
            'df3' : {
                    'name':'light2',
                    'score':97.95,
                    'df':df3 },
            'df4' : {
                    'name':'light3',
                    'score':97.95,
                    'df':df4 }
         }


# In[ ]:


df1.head()


# In[ ]:


# Making simple blendings of the models

isa_lg = 0
isa_hm = 0
print("Blending...\n")
for df in models.keys() : 
    isa_lg += np.log(models[df]['df'].is_attributed)
    isa_hm += 1/(models[df]['df'].is_attributed)
isa_lg = np.exp(isa_lg/4)
isa_hm = 1/isa_hm

print("Isa log\n")
print(isa_lg[:5])
print()
print("Isa harmo\n")
print(isa_hm[:5])


# In[ ]:


sub_log = pd.DataFrame()
sub_log['click_id'] = df1['click_id']
sub_log['is_attributed'] = isa_lg
sub_log.head()


# In[ ]:


sub_hm = pd.DataFrame()
sub_hm['click_id'] = df1['click_id']
sub_hm['is_attributed'] = isa_hm
sub_hm.head()


# In[ ]:


print("Writing...")
sub_log.to_csv('sub_log.csv', index=False, float_format='%.9f')
sub_hm.to_csv('sub_hm.csv', index=False, float_format='%.9f')