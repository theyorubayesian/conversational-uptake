"""Author: Dora Demszky

Predict uptake scores for utterance pairs, by running inference with an existing model checkpoint.

Usage:

python run_inference.py --data_file data/uptake_data.csv --speakerA student_text --speakerB teacher_text --output_col uptake_predictions --output predictions/uptake_data_predictions.csv

"""
import os
import sys
from typing import Dict, List, Union

import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import BertTokenizer, HfArgumentParser

from uptake.arguments import InferenceArguments
from uptake.inputs import BertInputBuilder, ModelInput
from uptake.models import MultiHeadModel
from uptake.utils import get_clean_text, get_num_words


def get_prediction(model: MultiHeadModel, instance: ModelInput, device: torch.device) -> Dict[str, torch.Tensor]:
    # TODO: This function should receive a batch of inputs
    instance["attention_mask"] = [1] * len(instance["input_ids"])
    for key in ["input_ids", "token_type_ids", "attention_mask"]:
        instance[key] = torch.tensor(instance[key]).unsqueeze(0).to(device)  # Batch size = 1

    output: Dict[str, torch.Tensor] = model(
        input_ids=instance["input_ids"],
        attention_mask=instance["attention_mask"],
        token_type_ids=instance["token_type_ids"],
        return_pooler_output=False
    )
    return output


def get_uptake_score(
    speakerA: Union[str, List[str]],
    speakerB: str,
    model: MultiHeadModel,
    input_builder: BertInputBuilder,
    max_length: int,
    remove_punct: bool = False,
    device: torch.device = "cuda:0",
) -> float:
    if isinstance(speakerA, str):
        textA = [get_clean_text(speakerA, remove_punct=remove_punct)]
    else:
        textA = [get_clean_text(x, remove_punct=remove_punct) for x in speakerA]
    
    textB = get_clean_text(speakerB, remove_punct=remove_punct)
    
    instance = input_builder.build_inputs(textA, textB, max_length=max_length, input_str=True)
    output = get_prediction(model, instance, device)
    score = softmax(output["nsp_logits"].squeeze(), dim=0).tolist()[1]
    return score


def run_batch_inference_batch(model: MultiHeadModel, batch):
    ...


def main():
    # parser = ArgumentParser()
    # parser.add_argument("--data_file", type=str, default="", help="Path or url of the dataset (csv).")
    # parser.add_argument("--speakerA", type=str, default="speakerA", help="Column indicating speaker A.")
    # parser.add_argument("--speakerB", type=str, default="speakerB", help="Column indicating speaker B (uptake is calculated for this speaker).")
    # parser.add_argument("--model_checkpoint", type=str,
    #                     default="checkpoints/Feb25_09-02-16_combined_education_dataset_02252021.json_6.25e-05_hist1_cand4_bert-base-uncased_ne1_nsp1",
    #                     help="Path, url or short name of the model")
    # parser.add_argument("--output_col", type=str, default="uptake_predictions",
    #                     help="Name of column for storing predictions.")
    # parser.add_argument("--output", type=str, default="",
    #                     help="Filename for storing predictions.")
    # parser.add_argument("--max_length", type=int, default=120, help="Maximum input sequence length")
    # parser.add_argument("--student_min_words", type=int, default=5, help="Maximum input sequence length")
    # parser.add_argument("--device", default="cuda:0", type=str)
    # args = parser.parse_args()
    parser = HfArgumentParser(InferenceArguments)
    args: InferenceArguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()

    print("Loading models...")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    input_builder = BertInputBuilder(tokenizer=tokenizer)
    uptake_model = MultiHeadModel.from_pretrained(args.model_name_or_path, head2size={"nsp": 2})
    uptake_model.to(args.device)

    utterances = pd.read_csv(args.data_file)
    print("EXAMPLES")
    for i, row in utterances.head().iterrows():
        print("speaker A: %s" % row[args.speakerA_column])
        print("speaker B: %s" % row[args.speakerB_column])
        print("----")

    print("Running inference on %d examples..." % len(utterances))
    uptake_model.eval()
    uptake_scores = []
    
    with torch.no_grad():
        for i, utt in utterances.iterrows():
            prev_num_words = get_num_words(utt[args.speakerA_column])
            if prev_num_words < args.speakerA_min_words:
                uptake_scores.append(None)
                continue
            
            uptake_score = get_uptake_score(
                speakerA=utt[args.speakerA_column],
                speakerB=utt[args.speakerB_column],
                model=uptake_model,
                device=args.device,
                input_builder=input_builder,
                max_length=args.max_seq_length
            )
            uptake_scores.append(uptake_score)

    utterances["predicted_uptake"] = uptake_scores
    utterances.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
