import torch
import pandas as pd
import argparse
import random
from transformers import AutoTokenizer, AutoModelForCausalLM


SYSTEM_PROMPT_FOR_DEBATES = """Generate a multi-turn dialogue between a debate decision maker who needs to take a decision about which
side wins a debate and a dialogue agent that has access to the arguments put forward by both the sides. The topic of the debate is {topic}. The arguments for both the sides are provided within []. 
The "For" args are :[{for_args}]. The "Against" args are: [{against_args}]. The dialogue should have minimum of 4 turns..."""

SYSTEM_PROMPT_FOR_METAREVIEWS = """Generate a multi-turn dialogue between a meta-reviewer and a dialogue agent for reviews about a paper. The title of the paper is: {paper}. The type of the paper is: {type}. The reviews are [Review 1:{review1}], [Review 2:{review2}] and [Review 3:{review3}]. ..."""

SYSTEM_PROMPT_FOR_HELPFUL_REVIEWS = """Generate a multi-turn dialogue between a buyer who wants to buy a product and a dialogue agent for reviews about that product. The product in discussion is {prod}. The reviews are [{reviews}]. The dialogue should have minimum of 4 turns..."""


def get_model_and_tokenizer(model_type):
    model_map = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "llama": "meta-llama/Llama-2-7b-chat-hf"
    }

    model_name = model_map.get(model_type.lower())
    if not model_name:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose from mistral, mixtral, llama.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", load_in_4bit=True
    )
    return tokenizer, model


def generate_response(model, tokenizer, prompt, device="cuda", max_new_tokens=1024):
    full_prompt = f"<s>[INST] {prompt.strip()} [/INST]"
    inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=0.7,
        do_sample=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.strip()


def write_output(base_path, domain, name, response, id_):
    output_path = f"{base_path}/src/output_dialogues/prompt_1/{domain}_{id_}.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(response.encode('ascii', 'ignore').decode('ascii'))


def generate_helpful_data(path, model, tokenizer):
    df = pd.read_csv(f'{path}/data/helpful_reviews.tsv', sep='\t')
    for _, row in df.iterrows():
        prompt = SYSTEM_PROMPT_FOR_HELPFUL_REVIEWS.format(prod=row["titles"], reviews=row["reviews"])
        response = generate_response(model, tokenizer, prompt)
        write_output(path, "helpful_reviews", row["titles"], response, row["id"])


def generate_debate_data(path, model, tokenizer):
    df = pd.read_csv(f'{path}/data/debates.tsv', sep='\t')
    for _, row in df.iterrows():
        prompt = SYSTEM_PROMPT_FOR_DEBATES.format(
            topic=row["topic"],
            for_args=row["for_args"],
            against_args=row["against_args"]
        )
        response = generate_response(model, tokenizer, prompt)
        write_output(path, "debates", row["topic"], response, row["ID"])


def generate_metareview_data(path, model, tokenizer):
    df = pd.read_csv(f'{path}/data/meta_reviews.tsv', sep='\t')
    for _, row in df.iterrows():
        prompt = SYSTEM_PROMPT_FOR_METAREVIEWS.format(
            paper=row["Paper"],
            type=row["Type"],
            review1=row["Review 1"],
            review2=row["Review 2"],
            review3=row["Review 3"]
        )
        response = generate_response(model, tokenizer, prompt)
        write_output(path, "meta_reviews", row["Paper"], response, row["ID"])


def main(args):
    tokenizer, model = get_model_and_tokenizer(args.model_type)

    if args.domain == "helpful_reviews":
        generate_helpful_data(args.path, model, tokenizer)
    elif args.domain == "debates":
        generate_debate_data(args.path, model, tokenizer)
    elif args.domain == "meta_reviews":
        generate_metareview_data(args.path, model, tokenizer)
    else:
        raise ValueError(f"Unknown domain: {args.domain}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Base path for data and outputs')
    parser.add_argument('--domain', type=str, choices=['helpful_reviews', 'debates', 'meta_reviews'], required=True)
    parser.add_argument('--model_type', type=str, choices=['mistral', 'mixtral', 'llama'], required=True)
    args = parser.parse_args()

    main(args)
