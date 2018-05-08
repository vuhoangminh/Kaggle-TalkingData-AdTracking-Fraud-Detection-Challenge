# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print("Reading the data...\n")
df1 = pd.read_csv('../output/input_ensemble/_w9812_submission_final4.csv')
df2 = pd.read_csv('../output/input_ensemble/_w9811_sub-it200102.csv')
# df3 = pd.read_csv('../output/input_ensemble/_w9797_final_gbdt_numleave31_maxdepth7_maxbin100_mininleaf128_round700_option18.csv')
# df4 = pd.read_csv('../output/input_ensemble/_w9798_final_gbdt_numleave63_maxdepth8_maxbin100_mininleaf100_round500_option18.csv')
# df5 = pd.read_csv('../output/input_ensemble/_w9796_final_gbdt_numleave31_maxdepth7_maxbin100_mininleaf100_round600_option18.csv')
df6 = pd.read_csv('../output/input_ensemble/_w9804_kaggle_rankavg.csv')
# df7 = pd.read_csv('../output/input_ensemble/_w9798_final_gbdt_numleave128_maxdepth16_maxbin512_mininleaf128_round400_option18.csv')
df8 = pd.read_csv('../output/input_ensemble/_w9800_kaggle_rankavg_4models.csv')
# df9 = pd.read_csv('../output/input_ensemble/_w9798_final_gbdt_numleave31_maxdepth7_maxbin100_mininleaf100_round500_option18.csv')
# df10 = pd.read_csv('../output/input_ensemble/_w9797_final_gbdt_numleave63_maxdepth8_maxbin100_mininleaf100_round500_option3.csv')
# df11 = pd.read_csv('../output/input_ensemble/_w9796_final_gbdt_numleave512_maxdepth64_maxbin1024_mininleaf128_round400_option3.csv')
# df12 = pd.read_csv('../output/input_ensemble/_w9796_final_gbdt_numleave512_maxdepth64_maxbin1024_mininleaf128_round400_option18.csv')
# df13 = pd.read_csv('../output/input_ensemble/_w9795_gdbt_sub_it_50percent_10_180000010_31_5_9795.csv')
# df14 = pd.read_csv('../output/input_ensemble/_w9795_final_gbdt_numleave31_maxdepth7_maxbin100_mininleaf100_round600_option3.csv')



models = {
    'df1': {
        'name': 'wordbatch_fm_ftrl',
        'score': 98.12,
        'df': df1 
    },
    'df2': {
        'name': 'sub_it7',
        'score': 98.11,
        'df': df2
    },  
    # 'df3': {
    #     'name': 'Krishna_s_CatBoost_1_1_CB_1_1',
    #     'score': 97.33,
    #     'df': df3
    # },
    # 'df4': {
    #     'name': 'submission_hm4',
    #     'score': 97.87,
    #     'df': df4
    # }, 
    # 'df5':{
    #     'name': 'submission_log4',
    #     'score': 1,
    #     'df': df5
    # }, 
    'df6':{
        'name': 'sub_it24',
        'score': 97.86,
        'df': df6
    },
    # 'df7':{
    #     'name': 'sub_log',
    #     'score': 97.80,
    #     'df': df7
    # },
    'df8':{
        'name': 'sub_hm',
        'score': 97.80,
        'df': df8
    }
    # 'df9':{
    #     'name': 'sub_hm',
    #     'score': 97.80,
    #     'df': df9
    # },
    # 'df10':{
    #     'name': 'sub_hm',
    #     'score': 97.80,
    #     'df': df10
    # },
    # 'df11':{
    #     'name': 'sub_hm',
    #     'score': 97.80,
    #     'df': df11
    # },
    # 'df12':{
    #     'name': 'sub_hm',
    #     'score': 97.80,
    #     'df': df12
    # },
    # 'df13':{
    #     'name': 'sub_hm',
    #     'score': 97.80,
    #     'df': df13
    # },
    # 'df14':{
    #     'name': 'sub_hm',
    #     'score': 97.80,
    #     'df': df14
    # }                    
}

count_models = len(models)  

isa_lg = 0
isa_hm = 0
isa_am = 0
isa_gm=0
print("Blending...\n")
for df in models.keys() : 
    isa_lg += np.log(models[df]['df'].is_attributed)
    isa_hm += 1/(models[df]['df'].is_attributed)
    isa_am += models[df]['df'].is_attributed
    isa_gm *= models[df]['df'].is_attributed
isa_lg = np.exp(isa_lg/count_models)
isa_hm = count_models/isa_hm
isa_am = isa_am/count_models
isa_gm = (isa_gm)**(1/count_models)

print("Isa log\n")
print(isa_lg[:count_models])
print()
print("Isa harmo\n")
print(isa_hm[:count_models])

sub_log = pd.DataFrame()
sub_log['click_id'] = df1['click_id']
sub_log['is_attributed'] = isa_lg
sub_log.head()

# sub_hm = pd.DataFrame()
# sub_hm['click_id'] = df1['click_id']
# sub_hm['is_attributed'] = isa_hm
# sub_hm.head()

# sub_fin0=pd.DataFrame()
# sub_fin0['click_id']=df1['click_id']
# sub_fin0['is_attributed']= (4*isa_lg+0.4*isa_hm+5.6*models['df1']['df'].is_attributed)/10

# sub_fin1=pd.DataFrame()
# sub_fin1['click_id']=df1['click_id']
# sub_fin1['is_attributed']= (4*isa_lg+6*models['df1']['df'].is_attributed)/10

# sub_fin2=pd.DataFrame()
# sub_fin2['click_id']=df1['click_id']
# sub_fin2['is_attributed']= (5*isa_lg+5*models['df1']['df'].is_attributed)/10

sub_fin3=pd.DataFrame()
sub_fin3['click_id']=df1['click_id']
sub_fin3['is_attributed']= (1*isa_lg+9*models['df1']['df'].is_attributed)/10

# sub_fin4=pd.DataFrame()
# sub_fin4['click_id']=df1['click_id']
# sub_fin4['is_attributed']= (3*isa_lg+0.1*isa_hm+6.9*models['df1']['df'].is_attributed)/10

# sub_fin5=pd.DataFrame()
# sub_fin5['click_id']=df1['click_id']
# sub_fin5['is_attributed']= (3*isa_lg+1.2*isa_hm+5.8*models['df1']['df'].is_attributed)/10



print(">> Writing...")
# sub_fin0.to_csv('40lg4hm56f1.csv', index=False, float_format='%.9f')
# print(">> Writing...")
# sub_fin1.to_csv('40lg60f1.csv', index=False, float_format='%.9f')
# print(">> Writing...")
# sub_fin2.to_csv('50lg50f1.csv', index=False, float_format='%.9f')
print(">> Writing...")
sub_fin3.to_csv('10lg90f1.csv', index=False, float_format='%.9f')
# print(">> Writing...")
# sub_fin4.to_csv('30lg1hm69f1.csv', index=False, float_format='%.9f')
# print(">> Writing...")
# sub_fin5.to_csv('30lg12hm58f1.csv', index=False, float_format='%.9f')
