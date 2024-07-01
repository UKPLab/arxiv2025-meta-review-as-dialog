import torch
from huggingface_hub import notebook_login
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import pandas as pd
import random
from transformers import AutoModelForCausalLM, AutoTokenizer



SYSTEM_PROMPT_for_debates = """Generate a multi-turn dialogue between a debate decision maker who needs to take a decision about which
                        side wins a debate and a dialogue agent that has access to the arguments put forward by both the sides. The topic of the debate is {topic}. The arguments for both the sides are provided within [].
                        The "For" args are :[{for_args}]. The "Against" args are: [{against_args}]. The dialogue should have minimum of 4 turns. You need to simulate both the decision maker and the dialogue agent. A decision maker has to reach a decision and explicitly tell which side wins the debate at the final turn. Every turn should have a decision maker utterance and a dialogue agent 
                        utterance. Every turn should start with either "Dialogue Agent" followed by their utterance or "Decision Maker" followed by their utterance. The decision maker does not have access to the arguments put forward by both the sides. The decision maker just knows the topic of the debate and should just rely on the dialogue agent to know about the arguments from both the sides. A decision 
                        maker mainly asks questions and the dialogue agent should just answer from the arguments and reply "I donot 
                        know" if there are some cases where its opinion/decision is sought. The dialogue agent has access to only the arguments and nothing else. The dialogue agent should never decide on who wins the debate and should always take a neutral stand when any opinion is sought. The topic of the debate is {topic}. The arguments for both the sides are provided within [].  The output should not have any copied sentences from the input but just the dialogue and no bullet points. 
                        The dialogue agent never answers in bullet points but always tries to summarize the answer"""

SYSTEM_PROMPT_for_metareviews = """Generate a multi-turn dialogue between a meta-reviewer and a dialogue agent for reviews about a paper.  The title of the paper is: {paper}. The type of the paper is: {type}. The reviews are [Review 1:{review1}], [Review 2:{review2}]
                        and [Review 3:{review3}]. The dialogue should have minimum of 4 turns. You need to simulate both a dialogue agent and the meta-reviewer. The meta-reviewer must reach a decision to "accept" or "reject" the paper at the final turn and must explicitly tell whether the paper is accepted or rejected. The dialogue ends with the meta-reviewer saying whether they accept or reject the paper. The reviews are provided within []. 
                        Every turn should have a meta-reviewer utterance and a dialogue agent utterance. Every turn should start with either "Dialogue Agent" followed by their utterance or "Meta Reviewer" followed by their utterance. The meta-reviewer just knows about the title of the paper does not have access to the reviews or the type of the paper and must rely on the dialogue agent
                        to know about the reviews and the type of the paper. The dialogue agent has access to only the reviews and type of the paper and nothing else. 
                        A meta-reviewer mainly asks questions and the dialogue agent should 
                        just answer from the reviews and reply "I donot know" if there are any questions that ask the dialogue agent's 
                        opinion. A dialogue agent should never recommend anything/ give any opinions/ decide anything for the paper. A dialogue agent has no confidence of its own.
                        A meta-reviewer can also ask follow-up questions and grill the dialogue agent for more information on the 
                        reviews. As a meta-reviewer you should also weigh the importance of the confidence of the reviewers while 
                        taking a decision.  The output should not have any copied sentences from the input but just the dialogue and no bullet points. The dialogue agent never answers in bullet points but always tries to summarize the answer.
      """

SYSTEM_PROMPT_for_helpful_reviews = """Generate a multi-turn dialogue between a buyer who wants to buy a product
                    and a dialogue agent for reviews about that product. The product in discussion is {prod}. The reviews are [{reviews}]. The dialogue should have minimum of 4 turns. You need to simulate both the buyer and the dialogue agent. The dialogue agent should always remain neutral and take a neutral stand in any case. The buyer should reach a decision to buy/not buy the paper
                    at the end. The buyer should explicitly tell whether they will buy the product or not. The reviews are provided within []. Every turn should have a buyer utterance and a dialogue agent utterance. Every turn should start with either "Dialogue Agent" followed by their utterance or "Buyer" followed by their utterance.  The
                    buyer does not have access to the reviews and has never read the reviews but the dialogue agent has access to the reviews and can answer any question about the reviews. The dialogue agent is not a seller of the product.  A buyer mainly
                    asks questions and the dialogue agent should just answer from the reviews and reply 'I donot know'
                    if the dialogue agent's opinion is sought. The dialogue agent has access to only the reviews and nothing else. The dialogue agent should not recommend/advise/decide anything about the product. The output should not have any copied sentences from the input but just the dialogue and no bullet points. The dialogue agent never answers in bullet points but always tries to summarize the answer."""

def main(args):
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
    device ="cuda"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
        device_map="auto", load_in_4bit=True)

    def generate_response(prompt: str, max_new_tokens: int = 4096) -> str:
        prompt = f"""<s>[INST]{prompt}[/INST]"""
        encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        model_inputs = encodeds.to(device)
        

        output = model.generate(**model_inputs,
                                max_length=max_new_tokens,
                                use_cache=True,
                                early_stopping=True,
                                bos_token_id=model.config.bos_token_id,
                                eos_token_id=model.config.eos_token_id,
                                pad_token_id=model.config.eos_token_id,
                                temperature=0.7,
                               do_sample=True)
        
        response = random.choice(tokenizer.batch_decode(output))
        print(response)
        return response

    def format_prompt(system_prompt: str) -> str:
        return f"""[INST]
        {system_prompt} [/INST]""".strip()

    
    
    

    def write_ouput(domain, topic, response, id):
        f = open(f'{args.path}/src/output_dialogues/prompt_1/mistral_raw/{domain}_{id}.txt','w')
        f.write(response.encode('ascii', 'ignore').decode('ascii'))
        f.close()

    def generate_helpful_data():
        
        help_df = pd.read_csv('{args.path}/data/helpful_reviews.tsv', sep='\t')
            
        for idx, rows in help_df.iterrows():
            SYSTEM_PROMPT = SYSTEM_PROMPT_for_helpful_reviews
            prompts = SYSTEM_PROMPT.format(prod = rows["titles"], reviews = rows["reviews"])
            debate_name = rows["titles"]
            response = generate_response(prompts)
            id = rows["id"]
            print(response)
            write_ouput('helpful_reviews',debate_name, response, id)            

    def generate_debate_data():
        debate_df = pd.read_csv(f'{args.path}/data/debates.tsv', sep='\t')

        for idx, rows in debate_df.iterrows():
            SYSTEM_PROMPT = SYSTEM_PROMPT_for_debates
            prompts = SYSTEM_PROMPT.format(topic = rows["topic"], for_args = rows["for_args"], against_args = rows["against_args"])
            debate_name = rows["topic"]
            response = generate_response(prompts)
            id = rows["ID"]
            print(response)
            write_ouput('debates',debate_name, response, id)     
            
    def generate_metareview_data():
        mr_df = pd.read_csv(f'{args.path}/data/meta_reviews.tsv', sep='\t')

        #print(mr_df.columns)

        for idx, rows in mr_df.iterrows():
           SYSTEM_PROMPT = SYSTEM_PROMPT_for_metareviews
           prompts = SYSTEM_PROMPT.format(paper = rows["Paper"], type = rows["Type"], review1 = rows["Review 1"], review2 = rows["Review 2"], review3 = rows["Review 3"])
           paper_name = rows["Paper"]
           response = generate_response(prompts)
           id = rows["ID"]
           print(response)
           write_ouput('meta_reviews',paper_name, response, id)     

    
    #generate_debate_data()
    generate_helpful_data()
    generate_metareview_data()


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    main(args)
