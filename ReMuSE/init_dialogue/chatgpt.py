import pandas as pd
import gzip
import json
import openai
from openai import AzureOpenAI
import argparse
from dialogue_utils import helpful_review_dialogues, debate_dialogues, meta_review_dialogues
from ..utils.gpt_utils import load_gpt


def chat_gpt_completion(output_path, prompts_list, prod_names, domain, ids, model_name):
    client = load_gpt()

    for prompt, prod_name, id_ in zip(prompts_list, prod_names, ids):
        try:
            messages = [
                {"role": "system", "content": prompt},
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                seed=123
            )
            output_text = response.choices[0].message.content
            print(output_text)

            output_file = f'{output_path}/{domain}_{id_}.txt'
            with open(output_file, 'w') as f:
                f.write(output_text)

        except openai.BadRequestError as e:
            print(f"BadRequestError for {domain}_{id_}: {e}")
        except Exception as e:
            print(f"Error for {domain}_{id_}: {e}")


def main(args):
    prompts_for_helpful_reviews = """Generate a multi-turn dialogue between a buyer ... The product in discussion is {prod}. The reviews are [{reviews}]"""
    prompts_for_debates = """Generate a multi-turn dialogue between a debate decision maker ... The "For" args are :[{for_args}]. The "Against" args are: [{against_args}]"""
    prompts_for_meta_reviews = """Generate a multi-turn dialogue between a meta-reviewer ... The title of the paper is: {paper}. The type of the paper is: {type}. The reviews are [Review 1:{review1}], [Review 2:{review2}] and [Review 3:{review3}]"""

    if args.domain == 'products':
        prompts, prod_names, prod_ids = helpful_review_dialogues('../data', prompts_for_helpful_reviews)

    elif args.domain == 'debates':
        prompts, prod_names, prod_ids = debate_dialogues('../data', prompts_for_debates)
    else:
        prompts, prod_names, prod_ids = meta_review_dialogues('../data', prompts_for_meta_reviews)

    chat_gpt_completion(
        output_path=args.out_path,
        prompts_list=prompts,
        prod_names=prod_names,
        domain=args.domain,
        ids=prod_ids,
        model_name=args.model_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, required=True, help='Path to store the output dialogues')
    parser.add_argument('--domain', type=str, default='meta-reviews', help='Domain for the dialogues: products, debates, or meta-reviews')
    parser.add_argument('--model_path', type=str, required=True, help='Model deployment name to use for the OpenAI completion')
    args = parser.parse_args()

    main(args)
