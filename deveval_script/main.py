from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import stream_jsonl,write_jsonl
from Context_Construct import Construct_context
import textwrap
from func_timeout import func_set_timeout
import func_timeout

model_name="deepseek-ai/deepseek-coder-1.3b-instruct"
dataset_path="/root/autodl-tmp/DevEval"
# result_path="/home/lisum/Code_generation/"

## 初始化模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
max_input_length = 10000
print("model load over")

###保留第一个函数
def get_first_function(code):
    lines = code.split('\n')
    first_function = []
    in_function = False
    for line in lines:
        if not in_function and line.startswith('def '):
            in_function = True
            first_function.append(line)
            continue
        if in_function:
            if line == '':
                 continue
            # 假设函数结束后有一个空行
            if line[0]!= ' ':
                break  
        if in_function:
            first_function.append(line)
    return '\n'.join(first_function)

def get_requirement(data):
    all_dependencys=data['dependency']
    project_name=data['namespace'].split(".")[0]
    completion_path=dataset_path+"/Source_Code/"+data['completion_path']
    
    dependency_path=dataset_path+"/Source_Code/"+data['completion_path']
    signature=""
    code=""
    with open(dependency_path,"r") as d:
        lines=d.readlines()
        for i in range(data['signature_position'][0]-1,data['signature_position'][1]):
            signature+=lines[i]
        for i in range(data['body_position'][0]-1,data['body_position'][1]):
            code+=lines[i]
    Functionality="\"\"\"\n"+data['requirement']['Functionality']+"\n"
    arguments=data['requirement']['Arguments'].split("\n")
    Arguments=""
    for argument in arguments:
        Arguments+=argument+"\n"
    requirement=signature.strip()+"\n"+textwrap.indent(Functionality+Arguments+"\"\"\""+"\n", '    ')
    return requirement

def get_query(namespace,code):
    all_data={data["namespace"]:data for data in stream_jsonl(dataset_path+"/data.jsonl")}
    data=all_data[namespace]
    query="#Please complete the input code, :param means parameter of this function, :return means return of this function.\n"
    project_name=data['namespace'].split(".")[0]
    completion_path=dataset_path+"/Source_Code/"+data['completion_path']
    all_dependencys=data['dependency']
    context=Construct_context(project_name,completion_path,all_dependencys)
    if(context!=""):
        context="#Here is the context:\n"+context
    dependency_path=dataset_path+"/Source_Code/"+data['completion_path']
    signature=""
    with open(dependency_path,"r") as d:
        lines=d.readlines()
        for i in range(data['signature_position'][0]-1,data['signature_position'][1]):
            signature+=lines[i]
    arguments=data['requirement']['Arguments'].split("\n")
    Arguments=""
    for argument in arguments:
        Arguments+=argument+"\n"
    input_ids = tokenizer(context, return_tensors="pt").to(model.device)
    input_len=input_ids["input_ids"].shape[1]
    if input_len > max_input_length:
        input_ids = {k: v[:, :max_input_length] for k, v in input_ids.items()}
    context=tokenizer.decode(input_ids["input_ids"][0],skip_special_tokens=True)
    query+=context+"#Here is the input code:\n"+code
    return query

@func_set_timeout(180) 
def generate(input_ids,requirement):
    input_len=input_ids["input_ids"].shape[1]
    output = model.generate(**input_ids, max_new_tokens=1500,do_sample=False)
    output=requirement+tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    return output



# output_file = "deveval_data.jsonl"

# for data in stream_jsonl(dataset_path + "/data.jsonl"):
#     requirement = get_requirement(data)
#     query = get_query(data["namespace"], requirement)

#     record = {
#         "namespace": data["namespace"],
#         "prompt": query
#     }
#     write_jsonl(output_file, [record], append=True)


# min_i = 1820
# i = 0
# for data in stream_jsonl(dataset_path + "/data.jsonl"):
#     i += 1
#     if i < min_i:
#         continue
    
#     requirement=get_requirement(data)
    
#     query=get_query(data["namespace"],requirement)
#     print(query)
    
#     input_ids = tokenizer(query, return_tensors="pt").to(model.device)
#     try:
#         output=generate(input_ids,requirement)
#     except func_timeout.exceptions.FunctionTimedOut:
#         output=""
        
#     result=get_first_function(output)
#     print(result)
#     write_jsonl("data_out_deepseek_without_fix.jsonl", [{"namespace":data["namespace"],"result":result}], append=True)