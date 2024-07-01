import pandas as pd
import gzip
import json
import os

import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
random.seed(42)






class LLama:
    def __init__(self):
        MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
        self.tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
        self.model = LlamaForCausalLM.from_pretrained(MODEL_NAME,
                device_map="auto", load_in_4bit=True)
        self.generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
        self.device='cuda'
        self.model = self.model.eval() 
    
    def generate_response(self,prompt: str, max_new_tokens: int = 4096) -> str:
        encoding = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **encoding,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                generation_config=self.generation_config,
            )
        answer_tokens = outputs[:, encoding.input_ids.shape[1] :]
        return self.tokenizer.decode(answer_tokens[0], skip_special_tokens=True)

#MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"


#tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
#model = LlamaForCausalLM.from_pretrained(MODEL_NAME,
#        device_map="auto", load_in_4bit=True)

 

class Mistral:
    def __init__(self, model_name):
        MODEL_NAME = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", load_in_4bit=True)
    
    def generate_response_mistral(self, prompt: str, device:str, max_new_tokens: int = 4096) -> str:
        prompt = f"""<s>[INST]{prompt}[/INST]"""
        encodeds = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        model_inputs = encodeds.to(device)
        

        output = self.model.generate(**model_inputs,
                                max_length=max_new_tokens,
                                use_cache=True,
                                early_stopping=True,
                                bos_token_id=self.model.config.bos_token_id,
                                eos_token_id=self.model.config.eos_token_id,
                                pad_token_id=self.model.config.eos_token_id,
                                temperature=0.7,
                               do_sample=True)
        
        response = random.choice(self.tokenizer.batch_decode(output))
        print(response)
        return response

def preprocess_dialogues_mistral(response):
    lines = response.split('[/INST][/INST]')
    feedback = lines[1]
    return feedback



#def generate_response(prompt: str, max_new_tokens: int = 4096) -> str:
#    encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
#    with torch.inference_mode():
#        outputs = model.generate(
#            **encoding,
#            max_new_tokens=max_new_tokens,
#            temperature=0.7,
#            generation_config=generation_config,
#        )
#    answer_tokens = outputs[:, encoding.input_ids.shape[1] :]
#    return tokenizer.decode(answer_tokens[0], skip_special_tokens=True)
   
def format_prompt(prompt: str) -> str:
    system_prompt = "You are an expert assistant that provides feedback to improve dialogues. The assistant only explicitly provides the feedback and not any example dialogues."
    return f"""[INST] <<SYS>> {system_prompt} <</SYS>> {prompt} [/INST]""".strip()  



def format_prompt_mistral(system_prompt: str) -> str:
    return f"""[INST]{system_prompt} [/INST]""".strip()

def create_dialogues(speaker, response, specificity):
    dialogues = []
    for s,r,sp in zip(speaker, response, specificity):
        dialogue = f'{s}: {r}, Score: {sp}'
        dialogues.append(dialogue)
    return '\n'.join(x for x in dialogues)

def get_data_specifity(metric_based_outputs, domain):
    df = pd.read_csv(metric_based_outputs, sep='\t', on_bad_lines='skip')
    df = df[df['Domain'] == domain]
    topic_names = df['topic_name'].unique()
    dialogues_parsed, knowledge_parsed, topics=[],[],[]
    for topic in topic_names:
        topic_df = df[df['topic_name'] == topic]
        knowledge = topic_df['knowledge'].tolist()[0]
        speaker = topic_df['speaker'].tolist()
        response = topic_df['response'].tolist()
        specificity = topic_df['specificty'].tolist()
        dialogues_parsed.append(create_dialogues(speaker, response, specificity))
        knowledge_parsed.append(knowledge)
        topics.append(topic)
    return dialogues_parsed, knowledge_parsed, topics


def create_dialogues_q2(speaker, response, f1_scores, nli_scores):
    dialogues = []
    for s,r,f1, nli in zip(speaker, response, f1_scores, nli_scores):
        dialogue = f'{s}: {r}, F1: {f1}, NLI: {nli}'
        dialogues.append(dialogue)
    return '\n'.join(x for x in dialogues)


def get_data_kprec(metric_based_outputs, domain):
    df = pd.read_csv(metric_based_outputs, sep='\t', on_bad_lines='skip')
    df = df[df['Domain'] == domain]
    topic_names = df['topic_name'].unique()
    dialogues_parsed, knowledge_parsed, topics=[],[],[]
    for topic in topic_names:
        topic_df = df[df['topic_name'] == topic]
        knowledge = topic_df['knowledge'].tolist()[0]
        speaker = topic_df['speaker'].tolist()
        response = topic_df['response'].tolist()
        kprec = topic_df['kprecision'].tolist()
        #nli = topic_df['nli'].tolist()
        dialogues_parsed.append(create_dialogues(speaker, response, kprec))
        knowledge_parsed.append(knowledge)
        topics.append(topic)
    return dialogues_parsed, knowledge_parsed, topics

def get_data_q2(metric_based_outputs, domain):
    df = pd.read_csv(metric_based_outputs, sep='\t', on_bad_lines='skip')
    df = df[df['Domain'] == domain]
    topic_names = df['topic_name'].unique()
    dialogues_parsed, knowledge_parsed, topics=[],[],[]
    for topic in topic_names:
        topic_df = df[df['topic_name'] == topic]
        knowledge = topic_df['knowledge'].tolist()[0]
        speaker = topic_df['speaker'].tolist()
        response = topic_df['response'].tolist()
        f1 = topic_df['f1'].tolist()
        nli = topic_df['nli'].tolist()
        dialogues_parsed.append(create_dialogues_q2(speaker, response, f1, nli))
        knowledge_parsed.append(knowledge)
        topics.append(topic)
    return dialogues_parsed, knowledge_parsed, topics

def get_feedback(knowledge, dialogues, domain, id, model, model_name, metric, path):
    #print('I am in the feedback loop')
    prompt= f"Given the knowledge and the dialogue, please provide actionable feedback to improve the dialogue. The feedback should just be for the overall dialogue and should start with 'Feedback :' and not have any sample dialogue utterances. Each turn of the dialogue is followed by a specificity score, the feedback should try to improve that score. Knowledge: {knowledge}\n. Dialogue: {dialogues}" 
    prompt_q2 = prompt= f"Given the knowledge and the dialogue, please provide actionable feedback to improve the dialogue. The feedback should just be for the overall dialogue and should start with 'Feedback :' and not have any sample dialogue utterances. Each turn of the dialogue is followed by a Q2 F1 score and Q2 NLI score, the feedback should try to improve those scores. Knowledge: {knowledge}\n. Dialogue: {dialogues}" 
    prompt_kprec= f"Given the knowledge and the dialogue, please provide actionable feedback to improve the dialogue. The feedback should just be for the overall dialogue and should start with 'Feedback :' and not have any sample dialogue utterances. Each turn of the dialogue is followed by a Knowledge Precision Score. the feedback should try to improve that score. Knowledge: {knowledge}\n. Dialogue: {dialogues}" 
    #print (prompt)
    if metric =='specifity':
        prompt = prompt
    if metric =='kprec':
        prompt = prompt_kprec
    if metric =='q2':
        prompt = prompt_q2

         #   if ("Feedback:") in line:
          #      idx = response.index(line)

        #data = ' '.join (x.strip('\n') for x in response.split() if x!='\n')
    if model_name =='llama':
        response = model.generate_response(format_prompt(prompt))
        print(response)
    if model_name =='mistral':
        output = model.generate_response_mistral(format_prompt_mistral(prompt), 'cuda')
        response = preprocess_dialogues_mistral(output)
        print(response)
    if model_name =='mixtral':
        output = model.generate_response_mistral(format_prompt_mistral(prompt), 'cuda')
        response = preprocess_dialogues_mistral(output)
        print(response)
    data = ' '.join (x.strip('\n') for x in response.split() if x!='\n')
    f = open(f'{path}/{domain}_{id}.txt','w')
    f.write(data)
    f.close()

def get_data(metric, metric_based_outputs, domain):
    if metric =='specifity':
        return get_data_specifity(metric_based_outputs, domain)
    if metric =='kprecision':
        return get_data_kprec(metric_based_outputs, domain)
    if metric =='q2':
        return get_data_q2(metric_based_outputs, domain)

def main(args):
    #print('hi')
    if args.model == 'mistral':
        model_m = Mistral('mistralai/Mistral-7B-Instruct-v0.1')
    if args.model == 'mixtral':
        model_m = Mistral('mistralai/Mixtral-8x7B-Instruct-v0.1')
    if args.model == 'llama':
        model_m = LLama()
    if args.model=='chatgpt':
        import openai
        from openai import AzureOpenAI
        

    metric_based_outputs = f'{args.path}/{args.prompt_type}/hallucination_{args.model}_outputs_{args.metric}.txt'
    for domain in ['debates', 'meta_reviews', 'helpful_reviews']:
    #for domain in ['helpful_reviews']:
        print(domain)
        dialogues, knowledge, topics = get_data(args.metric,metric_based_outputs, domain)

        new_path = f'{args.path}/feedback_{args.metric}{args.prompt_type}/{args.model}'
        os.makedirs(new_path, exist_ok=True)
        for (topic, dialogue, know) in zip(topics, dialogues, knowledge):
            get_feedback(know, dialogue, domain, topic, model_m, args.model, args.prompt_type, new_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='chatgpt')
    parser.add_argument('--prompt_type', type=str, default='prompt_1')
    parser.add_argument('--metric', type=str, default='specifity')
    parser.add_argument('--path', type=str, default='specifity')
    args = parser.parse_args()
    main(args)

