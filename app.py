from flask import Flask, request, jsonify, render_template
import re
# Use a pipeline as a high-level helper
from transformers import pipeline
import requests

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
headers = {"Authorization": "Bearer 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"}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    request_text = request.form['request_text']
    essay_text = request.form['essay_text']
    function_choice = int(request.form['function_choice'])

    if function_choice == 1:
        score, comment = process_chinese_essay(request_text, essay_text)
    elif function_choice == 2:
        score, comment = process_english_essay(request_text, essay_text)
    elif function_choice == 3:
        score, comment = process_chinese_translation(request_text, essay_text)
    elif function_choice == 4:
        score, comment = process_english_translation(request_text, essay_text)
    elif function_choice == 5:
        score, comment = process_text_expansion(request_text, essay_text)
    elif function_choice == 6:
        score, comment = process_text_abbreviation(request_text, essay_text)
    else:
        score, comment = 0, 'Invalid function choice'

    return jsonify({'score': score, 'comment': comment})

def callgpt(messages):
    response = requests.post(API_URL, headers=headers, json={"inputs": messages[0]['content']})
    output=response.json()
    return output[0]['generated_text']

# 处理中文作文输入，返回一个整数，两个字符，分别代表分数，评价，修改意见
def process_chinese_essay(title, content):
    message = [
       {"role":"system", "content":"你是一个语文作文老师，我是你的学生，你给我布置了这样一篇作文：\n\n%s\n\n你现在要给作文打分，打分需要依据的步骤如下：\
第一步:初步整体了解这篇文章,从思想内容和语言风格两个方面给一个初级分数；\
第二步:进一步了解文章结构,从组织结构和惯例两个方面给一个分数；\
第三步:更具体地了解词汇选择和句子流利度的细节；\
第四步:了解细节后,从组织结构和惯例再次评估结构；\
第五步:总结前面了解,从思想内容和语言风格再次整体评估。\
需要注意一点，这五个步骤中的两个方面是可以互相参考、相辅相成的。\
通过这五个步骤,给出最终的分数，并解释其原因。\
我的作文如下：\n\n%s\n\n以100分为满分，\
你一共给出两部分，首先给出总分，然后给出你的详细评价和改进意见，记住要一步一步地思考。"%(title, content)}
    ]
    pattern = '[\d+]+'
    result = callgpt(message)
    return re.findall(pattern, result)[0], result

# 处理英文作文输入
def process_english_essay(title, content):
    message = [
       {"role":"system", "content":"You are the essay teacher reviewing student essays. The writing prompt is that:\n\n%s\n\n\
The dimensions of scoring include six aspects: 1. Ideas and Content; 2. Organization; 3. Voice; 4. Word Choice; 5. Sentence Fluency; 6. Conventions. Our evaluation process has six steps:\
The first step is: Let's have an initial intuitive and overall understanding of this essay based on our impression from the aspects of ideas and content as well as voice.\
The second step is: Let's delve deeper and understand this essay from its structure in organization and conventions.\
The third step is: Let's be more specific and understand this essay from the details in word choice and fluency of sentences. \
The fourth step is: After understanding the details, let's re-evaluate the essay from organization and conventions in terms of its structure. \
The fifth step is: Finally, by comprehensively summarizing our previous understanding, let's evaluate the essay as a whole from the aspects of ideas and content as well as voice.\
It is worth noting that the two traits in each step are not isolated, but rather influence each other. \
Here is the article that needs you to provide scores for:\n\n%s\n\nPlease give your score.\
Using 100 points as the maximum score, please give your evaluation and suggestion. \
Attention! Firstly, give your total score only! Then give suggestions. You only need to give a total score, without scoring each trait separately. \
Let's evaluate the essay step by step."%(title, content)}
    ]
    pattern = '[\d+]+'
    result = callgpt(message)
    return re.findall(pattern, result)[0], result

# 处理中文翻译评价
def process_chinese_translation(chinese, english):
    message = [
       {"role":"system", "content":"你是一个英语老师，我是你的学生，你让我翻译这样一篇中文文章：\n\n%s\n\n这是我的翻译\
结果:\n\n%s\n\n请对我的作答进行评价与批改，以100分为满分，首先给出你的评分，然后给出你的评价和改进意见。"%(chinese, english)}
    ]
    pattern = '[\d+]+'
    result = callgpt(message)
    return re.findall(pattern, result)[0], result

# 处理英文翻译评价
def process_english_translation(english, chinese):
    message = [
       {"role":"system", "content":"你是一个英语老师，我是你的学生，你让我翻译这样一篇英语文章：\n\n%s\n\n这是我的翻译\
结果:\n\n%s\n\n请对我的作答进行评价与批改，以100分为满分，首先给出你的评分，然后给出你的评价和改进意见。"%(english, chinese)}
    ]
    pattern = '[\d+]+'
    result = callgpt(message)
    return re.findall(pattern, result)[0], result

# 处理文本扩写评价
def process_text_expansion(original, expanded):
    message = [
       {"role":"system", "content":"你是一个作文老师，我是你的学生，你让我将这段话进行扩展：\n\n%s\n\n这是我的作业：\n\n%s\n\n\
以100分为满分，首先给出你的评分，然后给出你的评价和改进意见。"%(original, expanded)}
    ]
    pattern = '[\d+]+'
    result = callgpt(message)
    return re.findall(pattern, result)[0], result

# 处理文本缩写评价
def process_text_abbreviation(original, abbreviated):
    message = [
       {"role":"system", "content":"你是一个作文老师，我是你的学生，你让我将这段话进行缩写：\n\n%s\n\n这是我的作业：\n\n%s\n\n\
以100分为满分，首先给出你的评分，然后给出你的评价和改进意见。"%(original, abbreviated)}
    ]
    pattern = '[\d+]+'
    result = callgpt(message)
    return re.findall(pattern, result)[0], result

if __name__ == '__main__':
    app.run(debug=True)

