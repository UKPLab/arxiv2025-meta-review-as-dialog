import pandas as pd
import os
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, GenerationConfig

random.seed(42)

# -----------------------------
# LLaMA Model Wrapper
# -----------------------------
class LLama:
    def __init__(self):
        MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
        self.tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
        self.model = LlamaForCausalLM.from_pretrained(MODEL_NAME,
                                                      device_map="auto",
                                                      load_in_4bit=True)
        self.generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
        self.device = 'cuda'
        self.model.eval()

    def generate_response(self, prompt: str, max_new_tokens: int = 8092) -> str:
        encoding = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(**encoding,
                                          max_new_tokens=max_new_tokens,
                                          temperature=0.7,
                                          generation_config=self.generation_config)
        answer_tokens = outputs[:, encoding.input_ids.shape[1]:]
        return self.tokenizer.decode(answer_tokens[0], skip_special_tokens=True)


# -----------------------------
# Mistral/Mixtral Model Wrapper
# -----------------------------
class Mistral:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          device_map="auto",
                                                          load_in_4bit=True)

    def generate_response_mistral(self, prompt: str, device: str, max_new_tokens: int = 4096) -> str:
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
        return response


# -----------------------------
# Helpers for prompts
# -----------------------------
def preprocess_dialogues_mistral(response):
    lines = response.split('[/INST][/INST]')
    return lines[1] if len(lines) > 1 else response

def format_prompt_feedback(prompt: str) -> str:
    system_prompt = "You are an expert assistant that provides feedback to improve dialogues. The assistant only explicitly provides the feedback and not any example dialogues."
    return f"""[INST] <<SYS>> {system_prompt} <</SYS>> {prompt} [/INST]"""

def format_prompt(prompt: str) -> str:
    system_prompt = "You are an expert assistant that can rewrite dialogues based on feedback. The assistant then explicitly outputs the new dialogue."
    return f"""[INST] <<SYS>> {system_prompt} <</SYS>> {prompt} [/INST]"""

def format_prompt_mistral(system_prompt: str) -> str:
    return f"""[INST]{system_prompt} [/INST]"""


# -----------------------------
# Data Handling
# -----------------------------
def create_dialogues(speaker, response, metrics_data, metrics_selected):
    dialogues = []
    for i in range(len(speaker)):
        dialogue = f"{speaker[i]}: {response[i]}"
        for metric in metrics_selected:
            dialogue += f", {metric.upper()}: {metrics_data[metric][i]}"
        dialogues.append(dialogue)
    return '\n'.join(dialogues)


def get_data(metric_based_outputs1, metric_based_outputs2, metric_based_outputs3, domain, metrics_selected):
    df1 = pd.read_csv(metric_based_outputs1, sep='\t', on_bad_lines='skip')
    df2 = pd.read_csv(metric_based_outputs2, sep='\t', on_bad_lines='skip')
    df3 = pd.read_csv(metric_based_outputs3, sep='\t', on_bad_lines='skip')

    df1 = df1[df1['Domain'] == domain]
    df2 = df2[df2['Domain'] == domain]
    df3 = df3[df3['Domain'] == domain]

    topic_names = df1['topic_name'].unique()
    dialogues_parsed, knowledge_parsed, topics = [], [], []

    for topic in topic_names:
        topic_df1 = df1[df1['topic_name'] == topic]
        topic_df2 = df2[df2['topic_name'] == topic]
        topic_df3 = df3[df3['topic_name'] == topic]

        knowledge = topic_df1['knowledge'].tolist()[0]
        speaker = topic_df1['speaker'].tolist()
        response = topic_df1['response'].tolist()

        # Collect all metrics
        metrics_data = {
            'specificity': topic_df1['specificty'].tolist(),
            'q2': topic_df2['f1'].tolist(),
            'kprec': topic_df3['kprecision'].tolist()
        }

        dialogues_parsed.append(create_dialogues(speaker, response, metrics_data, metrics_selected))
        knowledge_parsed.append(knowledge)
        topics.append(topic)

    return dialogues_parsed, knowledge_parsed, topics


# -----------------------------
# Feedback Generation
# -----------------------------
def get_feedback(prompt_type, model, model_name, epochs, domain, topic_id, knowledge, dialogues):
    if epochs == 1:
        prompt = f"Given the knowledge and the dialogue, please provide actionable feedback to improve the dialogue. The feedback should start with 'Feedback :' and not include sample dialogues. Each turn includes metrics for evaluation. Knowledge: {knowledge}\nDialogue: {dialogues}"
    else:
        prompt = f"Given the knowledge and the dialogue, please provide actionable feedback to improve the dialogue. The feedback should start with 'Feedback :' and not include sample dialogues. Knowledge: {knowledge}\nDialogue: {dialogues}"

    if model_name == 'llama':
        response = model.generate_response(format_prompt_feedback(prompt))
    else:
        output = model.generate_response_mistral(format_prompt_mistral(prompt), 'cuda')
        response = preprocess_dialogues_mistral(output)

    return response


# -----------------------------
# Refinement Step
# -----------------------------
def get_refinement(knowledge, dialogues, feedback, domain, topic_id, model, model_name, prompt_type):
    prompt = f"Given the feedback, knowledge, and dialogue, improve the dialogue. The output should be the new dialogue only. Knowledge: {knowledge}\nDialogue: {dialogues}\nFeedback: {feedback}"

    if model_name == 'llama':
        response = model.generate_response(format_prompt(prompt))
    else:
        output = model.generate_response_mistral(format_prompt_mistral(prompt), 'cuda')
        response = preprocess_dialogues_mistral(output)

    return response


# -----------------------------
# Main Execution
# -----------------------------
def main(args):
    metric_based_outputs3 = f'/storage/ukp/work/purkayastha/buy_this_paper/src/metrics/outputs/{args.prompt_type}/hallucination_{args.model}_outputs_kprecision.txt'
    metric_based_outputs2 = f'/storage/ukp/work/purkayastha/buy_this_paper/src/metrics/outputs/{args.prompt_type}/hallucination_{args.model}_outputs_q2.txt'
    metric_based_outputs1 = f'/storage/ukp/work/purkayastha/buy_this_paper/src/metrics/outputs/{args.prompt_type}/hallucination_{args.model}_outputs_specifity.txt'

    metrics_selected = [m.strip().lower() for m in args.metrics.split(',')]
    print(f"Using metrics: {metrics_selected}")

    # Initialize model
    if args.model == 'mistral':
        model_m = Mistral('mistralai/Mistral-7B-Instruct-v0.1')
    elif args.model == 'mixtral':
        model_m = Mistral('mistralai/Mixtral-8x7B-Instruct-v0.1')
    else:
        model_m = LLama()

    # Process domains
    for domain in ['debates', 'meta_reviews', 'helpful_reviews']:
        print(f"Processing domain: {domain}")
        dialogues, knowledge, topics = get_data(metric_based_outputs1,
                                               metric_based_outputs2,
                                               metric_based_outputs3,
                                               domain,
                                               metrics_selected)

        new_path = f'/storage/ukp/work/purkayastha/buy_this_paper/src/refinement_all_scores/{args.prompt_type}/{args.model}/{args.epochs}'
        os.makedirs(new_path, exist_ok=True)

        for epoch in range(args.epochs):
            all_feedbacks = []
            for topic, feed, know in zip(topics, dialogues, knowledge):
                feedback = get_feedback(args.prompt_type, model_m, args.model, epoch, domain, topic, know, feed)
                all_feedbacks.append(feedback)

            for topic, dialogue, know, feedback in zip(topics, dialogues, knowledge, all_feedbacks):
                refined_dialogue = get_refinement(know, dialogue, feedback, domain, topic, model_m, args.model, args.prompt_type)
                with open(f'{new_path}/{domain}_{topic}.txt', 'w') as f:
                    f.write(refined_dialogue)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'mistral', 'mixtral'])
    parser.add_argument('--prompt_type', type=str, default='prompt_1')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--metrics', type=str, default='kprec,q2,specificity', help="Comma-separated list of metrics to include: kprec,q2,specificity")
    args = parser.parse_args()
    main(args)
