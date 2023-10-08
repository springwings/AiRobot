# -*- coding: utf-8 -*-
# python 3.6
"""
ChatGpt

Author: chengwen
Modifier:
date:   2023/3/13 上午11:45
Description:
"""
from http import HTTPStatus

import dashscope
from dashscope import Generation

dashscope.api_key = "aliyun code"

class ChatGpt(): 
   

    def __init__(self):
        pass
 	
    def predict(self,messages):
        gen = Generation()
        response = gen.call(
           Generation.Models.qwen_turbo,
           messages=messages,
           result_format='message', 
        ) 
        if response.status_code == HTTPStatus.OK:   
           return response  
        else:
           print('Request id: %s, Status code: %s, error code: %s, error message: %s'%(
              response.request_id, response.status_code, 
              response.code, response.message
        ))     
        
    def __call__(self,prompt):
        
        response = self.predict(prompt)
        try: 
            return response.output.choices[0]['message']['content']
        except:
            return ""
             
            
