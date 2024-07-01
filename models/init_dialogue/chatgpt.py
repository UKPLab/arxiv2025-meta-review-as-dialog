import pandas as pd
import gzip
import json
import openai
from openai import AzureOpenAI



def chat_gpt_completion(prod_prompts, prod_names, domain, ids):
    for prompts,prods,id in zip(prod_prompts,prod_names,ids):
        try:
            curr_response = client.chat.completions.create(
            model=deployment_name, 
            messages=[
                {"role": "system", "content": prompts},
            ]
            )
        except openai.BadRequestError as e:
            f1.write(f'{domain}_{id}\n')
            print(f'{domain}_{id}')
        print(curr_response.choices[0].message.content)
       



def main():
    
    prompts_for_helpful_reviews = """Generate a multi-turn dialogue between a buyer who wants to buy a product
                    and a dialogue agent for reviews about that product. Every turn should start with either "Dialogue Agent" followed by their utterance or "Buyer" followed by their utterance. The dialogue agent should always remain neutral and take a neutral stand in any case. The buyer should reach a decision to buy/not buy the paper
                    at the final turn. The buyer should explicitly tell whether they will buy the product or not. The reviews are provided within []. You need to simulate both the
                    buyer and the dialogue agent. Every turn should have a buyer utterance and a dialogue agent utterance.  The
                    buyer does not have access to the reviews and has never read the reviews but the dialogue agent has access to the reviews and can answer any question about the reviews. The dialogue agent is not a seller of the product.  A buyer mainly
                    asks questions and the dialogue agent should just answer from the reviews and reply 'I donot know'
                    if the dialogue agent's opinion is sought. The dialogue agent has access to only the reviews and nothing else. The dialogue agent should not recommend/advise/decide anything about the product. The product in discussion is {prod}. The reviews are [{reviews}]"""
    
    prompts_for_debates = """Generate a multi-turn dialogue between a debate decision maker who needs to take a decision about which
                        side wins a debate and a dialogue agent that has access to the arguments put forward by both the sides.  
                        Every turn should start with either "Dialogue Agent" followed by their utterance or "Decision Maker" followed by their utterance. A decision maker has to reach a decision and explicitly tell which side wins the debate at the final turn. You need to simulate both the 
                        decision maker and the dialogue agent. Every turn should have a decision maker utterance and a dialogue agent 
                        utterance.  The decision maker does not have access to the arguments put forward by both the sides. The decision maker just knows the topic of the debate and should just rely on the dialogue agent to know about the arguments from both the sides. A decision 
                        maker mainly asks questions and the dialogue agent should just answer from the arguments and reply "I donot 
                        know" if there are some cases where its opinion/decision is sought. The dialogue agent has access to only the arguments and nothing else. The dialogue agent should never decide on who wins the debate and should always take a neutral stand when any opinion is sought. The topic of the debate is {topic}. The arguments for both the sides are provided within [].
                        The "For" args are :[{for_args}]. The "Against" args are: [{against_args}]"""

    prompts_for_meta_reviews = """Generate a multi-turn dialogue between a meta-reviewer and a dialogue agent for reviews about a paper. Every turn should start with either "Dialogue Agent" followed by their utterance or "Meta Reviewer" followed by their utterance
                        The meta-reviewer must reach a decision to "accept" or "reject" the paper at the final turn and must explicitly tell whether the paper is accepted or rejected. The dialogue ends with the meta-reviewer saying whether they accept or reject the paper. The reviews are provided within []. 
                        You need to simulate both the meta-reviewer and the dialogue agent. Every turn should have a meta-reviewer 
                        utterance and a dialogue agent utterance. The meta-reviewer just knows about the title of the paper does not have access to the reviews or the type of the paper and must rely on the dialogue agent
                        to know about the reviews and the type of the paper. The dialogue agent has access to only the reviews and type of the paper and nothing else. 
                        A meta-reviewer mainly asks questions and the dialogue agent should 
                        just answer from the reviews and reply "I donot know" if there are any questions that ask the dialogue agent's 
                        opinion. A dialogue agent should never recommend anything/ give any opinions/ decide anything for the paper. A dialogue agent has no confidence of its own.
                        A meta-reviewer can also ask follow-up questions and grill the dialogue agent for more information on the 
                        reviews. As a meta-reviewer you should also weigh the importance of the confidence of the reviewers while 
                        taking a decision. The title of the paper is: {paper}. The type of the paper is: {type}. The reviews are [Review 1:{review1}], [Review 2:{review2}]
                        and [Review 3:{review3}]"""


    def helpful_review_dialogues():
        help_df = pd.read_csv(f'{args.path}/helpful_reviews.tsv', sep='\t')
        prod_prompts, prod_names,ids = [],[],[]

        for idx, rows in help_df.iterrows():
            prod_prompts.append(prompts_for_helpful_reviews.format(prod = rows["titles"], reviews = rows["reviews"]))
            prod_names.append(rows["titles"])
            ids.append(rows["id"])

        print(f"Prompt:{prod_prompts[0]}\n")
        print("------------------------------------------------\n")

        print("ChatGPT response:\n")
        print("------------------------------------------------\n")
        chat_gpt_completion(prod_prompts, prod_names, 'helpful_reviews', ids)

    #---------------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------------

    def debate_dialogues():


        debate_df = pd.read_csv(f'{args.path}/debates.tsv', sep='\t')
        
        debate_prompts, debate_names, ids=[], [],[]
        
        for idx, rows in debate_df.iterrows():
            debate_prompts.append(prompts_for_debates.format(topic = rows["topic"], for_args = rows["for_args"], against_args = rows["against_args"]))
            debate_names.append(rows["topic"])
            ids.append(rows["ID"])
        chat_gpt_completion(debate_prompts, debate_names, 'debates', ids)


    #---------------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------------

    def meta_review_dialogues():
        mr_df = pd.read_csv(f'{args.path}/meta_reviews.tsv', sep='\t')

        mr_prompts, paper_names, ids = [],[],[]

        for idx, rows in mr_df.iterrows():
            mr_prompts.append(prompts_for_meta_reviews.format(paper = rows["Paper"], type = rows["Type"], review1 = rows["Review 1"], review2 = rows["Review 2"], review3 = rows["Review 3"]))
            paper_names.append(rows["Paper"])
            ids.append(rows["ID"])
        
        chat_gpt_completion(mr_prompts, paper_names, 'meta_reviews', ids)

    helpful_review_dialogues()
    debate_dialogues()
    meta_review_dialogues()

if __name__=="__main__":
 parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    main(args)
    main()
