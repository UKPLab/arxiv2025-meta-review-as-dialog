# Decision-Making with Deliberation: Meta-reviewing as a Document-grounded Dialogue

[![Arxiv](https://img.shields.io/badge/Arxiv-2409.19505-red?style=flat&logo=arxiv&logoColor=white)](https://www.arxiv.org/abs/2508.05283)
[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
>  **Abstract**
>
> Meta-reviewing is a pivotal stage in the peer-review process, serving as the final step in determining whether a paper is recommended for acceptance. Prior research on meta-reviewing has treated this as a summarization problem over review reports. However, complementary to this perspective, meta-reviewing is a decision-making process that requires weighing reviewer arguments and placing them within a broader context. Prior research has demonstrated that decision-makers can be effectively assisted in such scenarios via dialogue agents. In line with this framing,
we explore the practical challenges for realizing dialog agents that can effectively assist meta-reviewers. Concretely, we first address the issue of data scarcity for training dialogue agents by generating synthetic data using Large Language Models (LLMs) based on a self-refinement strategy to improve the relevance of these dialogues to expert domains. Our experiments demonstrate that this method produces higher-quality synthetic data and can serve as a valuable resource towards training meta-reviewing assistants. Subsequently, we utilize this data to train dialogue agents tailored for meta-reviewing and find that these agents outperform *off-the-shelf* LLM-based assistants for this task. Finally, we apply our agents in real-world meta-reviewing scenarios and confirm their effectiveness in enhancing the efficiency of meta-reviewing.
>
This repository contains the code to reproduce the ReMuSE experiments on different domains along with the dialogue agent setups that we used in our paper. 

<p align="center">
<img src="assets/Fig1_v2.png" width="500">
</p>


Contact person: [Sukannya Purkayastha](mailto:sukannya.purkayastha@tu-darmstadt.de)

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.



## Setup and WorkFlow
For running the experiments, one needs to install necessary packages that we provide in the ``requirements.txt`` file as below:
>
```bash
$ conda create -n decision_making python=3.10
$ conda activate decision_making
$ pip install -r requirements.txt
```
>

## ReMuSE experiments
ReMuSE has three phases and we provide the steps to run each of them below:

### Initial Dialogue Generation

>
    export out_path = 'output_llama_7B'

    python ReMuSE/init_dialogue/open_source_llms.py \
    --out_path $out_path \
    --model_path 'meta-llama/Llama-2-13b-chat-hf' \
    --domain 'meta_reviews'
>
In our paper, we performed experiments with the following models:
| Name     | Sizes | 🤗 model links   |
| :---: | :---: | :---: |
| LLaMa 2 chat    |  7B, 13B | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) |
| Mistral Instruct    | 7B   | [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)  |
| Mixtral Instruct    | 7B   | [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) |

To run the other models, one needs to just change the ``model_path`` from the above tables. The ``domain`` can be either of ``meta_reviews /debates/ helpful_reviews``. The ``out_path`` should be the path where one wants to save the genertaed dialogues. We also perfrom experiments with  GPT 3.5 (ChatGPT), that can be run with the follwoing command:
>
export out_path = 'output_chatgpt'
>
>
    python ReMuSE/init_dialogue/chatgpt.py \
    --out_path $out_path \
    --model_path 'gpt-3.5-turbo-0125'
>

### Evaluation
The dialogues generated in the previous step, need to be evaluated with different factuality and specificity metrics. We need to first create the dialogue in the format for the evaluation metrics:
>
>
    python metrics/create_hallucination_format.py \
    --model 'llama7B' \
    --path 'output_llama_7B' \
    --output_path 'metrics_augmented' \
    --domain 'meta_reviews'
>
>
The ``model`` here is the short url for the models used earlier. The name is used to just store outputs in the file named: ``{args.output_path}/hallucination_outputs_{args.model}"``. The ``output_path`` is therefore any convenient path to store the output. The ``path`` should be the path where the generated dialogues are, e.g., ``output_llama_7B``. The ``domain`` argument is same as before.

After creating the format for the data, we then evaluate each of the utterances in the dialogue using different metrics.

For ``K-Prec``, ``Q2``, we adapt the code from the excellent [instruct-qa repository](https://github.com/McGill-NLP/instruct-qa). In connection to our repository, it needs to be run as follows:
>
    python metrics/evaluate_responses.py \
    --path $data_path \
    --metric q2 \
    --model llama7B \
    --output_path $out
>
The ``path`` argument takes the path where we stored the formatted outputs before (essentially the ``$output_path`` in the previous code). The ``metric`` can be one of ``faithcritic, q2, kbert, kprecision, faithcritic_labels``. The ``model`` argument is the shortened version as in the previous code (e.g., llama7B). The ``output_path`` is the path where one wants to store the outputs from this function.

For ``Specificity``, we adapt the code for [Speciteller](https://github.com/jjessyli/speciteller), as below:
>
    python metrics/speciteller/python3_code/speciteller.py \
    --path refinement_kprec_spec \
    --filename "hallucination_outputs_chatgpt.txt" \
    --model "chatgpt" 
> 

The ``path`` argument should be the ``$out`` in the previous code. The      ``filename`` is the output filename. The  ``model`` argument is the shortened model name that is used throughout.  


### Feedback Generation and Refinement
To generate the feedback and refinement, we use the following code:

>
    python ReMuSE/feedback_and_refinement/refinement_pipeline.py \
    --model mixtral \
    --metrics kprec,q2,specificity \
    --epochs 2
>
The ``model`` argument is the shortened name for the model taht we have used throughout. The ``metrics`` can be any combination of `k-prec, q2 and specificity'. The ``epochs`` argument can be any number.

One can iteratively run the codes within the ``Evaluation`` section again to get the scores for the refined dialogues.
## Citation

```bib
@misc{purkayastha2025decisionmakingdeliberationmetareviewingdocumentgrounded,
      title={Decision-Making with Deliberation: Meta-reviewing as a Document-grounded Dialogue}, 
      author={Sukannya Purkayastha and Nils Dycke and Anne Lauscher and Iryna Gurevych},
      year={2025},
      eprint={2508.05283},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.05283}, 
}
```
