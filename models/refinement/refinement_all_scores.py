import pandas as pd
import gzip
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import torch
import random

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
    
    def generate_response(self,prompt: str, max_new_tokens: int = 8092) -> str:
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


def format_prompt(prompt: str) -> str:
    system_prompt = "You are an expert assistant that can rewrite dialogues based on feedback. The assistant then explicitly outputs the new dialogue."
    return f"""[INST] <<SYS>> {system_prompt} <</SYS>> {prompt} [/INST]""".strip()  

def format_prompt_mistral(system_prompt: str) -> str:
    return f"""[INST]{system_prompt} [/INST]""".strip()



def create_dialogues(speaker, response):
    dialogues = []
    for s,r in zip(speaker, response):
        dialogue = f'{s}: {r}'
        dialogues.append(dialogue)
    return '\n'.join(x for x in dialogues)


def format_prompt_feedback(prompt: str) -> str:
    system_prompt = "You are an expert assistant that provides feedback to improve dialogues. The assistant only explicitly provides the feedback and not any example dialogues."
    return f"""[INST] <<SYS>> {system_prompt} <</SYS>> {prompt} [/INST]""".strip()

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
    system_prompt = "You are an expert assistant that can rewrite dialogues based on feedback. The assistant then explicitly outputs the new dialogue."
    return f"""[INST] <<SYS>> {system_prompt} <</SYS>> {prompt} [/INST]""".strip()  

def get_feedback(prompt, model, model_name, epochs, domain, id, knowledge, dialogues):
    if epochs ==1:
        prompt= f"Given the knowledge and the dialogue, please provide actionable feedback to improve the dialogue. The feedback should just be for the overall dialogue and should start with 'Feedback :' and not have any sample dialogue utterances. Each turn of the dialogue is followed by a specificity score, Q_F1 score, Q_NLI score and Kprec _Score; the feedback should try to improve all the scores. Knowledge: {knowledge}\n. Dialogue: {dialogues}" 
    else:
        prompt= f"Given the knowledge and the dialogue, please provide actionable feedback to improve the dialogue. The feedback should just be for the overall dialogue and should start with 'Feedback :' and not have any sample dialogue utterances. Knowledge: {knowledge}\n. Dialogue: {dialogues}" 
    if model_name =='llama':
        response = model.generate_response(format_prompt_feedback(prompt))
        print(response)
    if model_name =='mistral':
        output = model.generate_response_mistral(format_prompt_mistral(prompt), 'cuda')
        response = preprocess_dialogues_mistral(output)
        print(response)
    if model_name =='mixtral':
        output = model.generate_response_mistral(format_prompt_mistral(prompt), 'cuda')
        response = preprocess_dialogues_mistral(output)
        print(response)
    return response

def create_dialogues(speaker, response, specificity, q_f1, q_nli, kprec):
    dialogues = []
    for s,r,sp,f1,nli,prec in zip(speaker, response, specificity, q_f1, q_nli, kprec):
        dialogue = f'{s}: {r}, Spec: {sp}, Q_F1: {f1}, Q_NLI: {nli}, Kprec: {prec}'
        dialogues.append(dialogue)
    return '\n'.join(x for x in dialogues)

def get_data(metric_based_outputs1, metric_based_outputs2, metric_based_outputs3, domain):
    df1 = pd.read_csv(metric_based_outputs1, sep='\t', on_bad_lines='skip')
    df2 = pd.read_csv(metric_based_outputs2, sep='\t', on_bad_lines='skip')
    df3 = pd.read_csv(metric_based_outputs3, sep='\t', on_bad_lines='skip')
    df1 = df1[df1['Domain'] == domain]
    df2 = df2[df2['Domain'] == domain]
    df3 = df3[df3['Domain'] == domain]
    topic_names = df1['topic_name'].unique()
    dialogues_parsed, knowledge_parsed, topics=[],[],[]
    for topic in topic_names:
        topic_df1 = df1[df1['topic_name'] == topic]
        topic_df2 = df2[df2['topic_name'] == topic]
        topic_df3 = df3[df3['topic_name'] == topic]
        knowledge = topic_df1['knowledge'].tolist()[0]
        speaker = topic_df1['speaker'].tolist()
        response = topic_df1['response'].tolist()
        specificity = topic_df1['specificty'].tolist()
        q_f1 = topic_df2['f1'].tolist()
        q_nli = topic_df2['nli'].tolist()
        kprec = topic_df3['kprecision'].tolist()
        dialogues_parsed.append(create_dialogues(speaker, response, specificity, q_f1, q_nli, kprec))
        knowledge_parsed.append(knowledge)
        topics.append(topic)
    return dialogues_parsed, knowledge_parsed, topics

def get_refinement(knowledge, dialogues, feedback, domain, id, model, model_name, prompt_type):
    #print('I am in the feedback loop')
    prompt= f"Given the feedback, knowledge and the dialogue, improve the dialogue. The new dialogue should improve the previous dialogue based on the feedback provided. Knowledge: {knowledge}\n. Dialogue: {dialogues}\n. Feedback:{feedback}. The output should just be the new dialogue" 
    #print (prompt)

    if model_name =='llama':
        response = model.generate_response(format_prompt(prompt))
        print(response)
        data = response
    if model_name =='mistral':
        output = model.generate_response_mistral(format_prompt_mistral(prompt), 'cuda')
        response = preprocess_dialogues_mistral(output)
        data = response
        print(response)
    if model_name =='mixtral':
        output = model.generate_response_mistral(format_prompt_mistral(prompt), 'cuda')
        response = preprocess_dialogues_mistral(output)
        print(response)
        data = response
   
    return data

def main(args):


    if args.model == 'mistral':
        model_m = Mistral('mistralai/Mistral-7B-Instruct-v0.1')
    if args.model == 'mixtral':
        model_m = Mistral('mistralai/Mixtral-8x7B-Instruct-v0.1')
    if args.model == 'llama':
        model_m = LLama()
    for domain in ['debates', 'meta_reviews', 'helpful_reviews']:
        print(domain)
        dialogues, knowledge, topics = get_data(args.metric_based_outputs1,args.metric_based_outputs2, args.metric_based_outputs3, domain)
        new_path = f'args.path/{args.prompt_type}/{args.model}/{args.epochs}'
        os.makedirs(new_path, exist_ok=True)
        for epoch in range(args.epochs):
            all_feedbacks = []
            for (topic, feed, know) in zip(topics, dialogues, knowledge):
                feedback = get_feedback(args.prompt_type, model_m, args.model, epoch, domain, topic, know, feed)
                all_feedbacks.append(feedback)
            feedbacks = all_feedbacks
            
            for (topic, dialogue, know, feedback) in zip(topics, dialogues, knowledge, feedbacks):
                dialogs = get_refinement(know, dialogue, feedback, domain, topic, model_m, args.model, args.prompt_type)
                f = open(f'{new_path}/{domain}_{topic}.txt','w')
                f.write(dialogs)
                f.close()
            
         


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama')
    parser.add_argument('--prompt_type', type=str, default='prompt_3')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--metric_based_outputs1', type=str)
    parser.add_argument('--metric_based_outputs2', type=str)
    parser.add_argument('--metric_based_outputs3', type=str)
    args = parser.parse_args()
    main(args)

