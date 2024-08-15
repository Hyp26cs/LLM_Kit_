import requests
import openai
import os
import numpy as np
import torch
import ast

#dataset
train_file = 'processed/res_train.npy'

print('Loading dataset......')
train_data = np.load(train_file,allow_pickle=True)
assert(train_data.shape[1]%3==1)
train_sen,train_label = train_data[:,0],torch.LongTensor(train_data[:,1:].astype('int8'))
aspects_num = train_label.shape[1]
print('Aspect number:{}'.format(aspects_num))

#ACSA
def generate_text(key,text):
    openai.api_key = os.getenv('OPENAI_KEY', default=key)
    
    #prompt
    prompt=f"""According to the following sentiment elements definition:
              - The “aspect category” refers to the category that aspect belongs to, and the available categories include: 
              “FOOD#QUALITY”, “FOOD#STYLE_OPTIONS”, “RESTAURANT#GENERAL”, “SERVICE#GENERAL”, “AMBIENCE#GENERAL”, “DRINKS#STYLE_OPTIONS”, “FOOD#PRICES”, 
              “RESTAURANT#PRICES”,“LOCATION#GENERAL”, “DRINKS#QUALITY”, “RESTAURANT#MISCELLANEOUS”,“DRINKS#PRICES”.
              - The “sentiment polarity” refers to the degree of positivity, negativity or neutrality expressed in the opinion towards a particular aspect or feature of a product or
              service, and the available polarities include: “positive”, “negative” and “neutral”. “neutral” means mildly positive or mildly negative. Quadruplets with objective
              sentiment polarity should be ignored.
              -Recognize all sentiment elements with their corresponding  aspect categories, sentiment polarity in the given input text \"{text}\".
              Provide your response in the format of a Python list of lists without your any analysis: ’ [[ “aspect category”, “sentiment polarity”], ...]’.
              Note that “, ...” indicates that there might be more lists in the list if applicable and must not occur in the answer. Ensure there is no additional text in the response.
              If the given text does not provide specific aspect categories,just return'[]'."""
    
    messages = [
        {"role": "system", "content": "You're a user review sentiment analyst."},
        {"role": "user", "content": prompt},
    ]

    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        # model="gpt-4-32k-0613",
        model="gpt-4",
        messages=messages,
        temperature=1.0,
    )
    result = ''
    
    for choice in response.choices:
        #print('content:',choice.message.content,'\n')
        #print(type(choice.message.content))
        result += choice.message.content
    lst = ast.literal_eval(result)
    return lst

key = "Your OpenAi Key"


#构造预测结果
asp2id = {"FOOD#QUALITY": 1, "FOOD#STYLE_OPTIONS": 2, "RESTAURANT#GENERAL": 3, "SERVICE#GENERAL": 4, "AMBIENCE#GENERAL": 5,
        "DRINKS#STYLE_OPTIONS": 6, "FOOD#PRICES": 7, "RESTAURANT#PRICES": 8, "LOCATION#GENERAL": 9, "DRINKS#QUALITY": 10, "RESTAURANT#MISCELLANEOUS": 11, "DRINKS#PRICES": 12}
sen2id ={"negative":0,"neutral":1,"positive":2}
data = np.load("res_pre.npy",allow_pickle=True)


for i in range(len(train_sen)):
    print(i,"\n")
    lst = generate_text(key,train_sen[i]) 
    #print(type(lst))
    if lst != '[]':
        for lt in lst:
           asp = asp2id[lt[0]]
           sen = sen2id[lt[1]]
           data[i][asp-1] = sen
np.save("res_pre.npy",data)





