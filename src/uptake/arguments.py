from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceArguments:
    data_file: Optional[str] = field(
        default=None, metadata={"help": "Path or url of the dataset (csv)."}
    )
    speakerA_column: str = field(
        default="speakerA", metadata={"help": "Column indicating speaker A."}
    )
    speakerB_column: str = field(
        default="speakerB", metadata={"help": "Column indicating speaker B."}
    )
    speakerA_min_words: int = field(
        default=5, 
        metadata={
            "help": (
                "Minimum number of words in speakerA text. "
                "Uptake is not calculated if not enough words"
            )
        }
    )
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path, url or short name of the model"}
    )
    device: str = field(
        default="cuda:0", metadata={"help": "Compute resource to use: cpu or cuda"}
    )
    batch_size: int = field(
        default=8, metadata={"help": "Batch size for inference"}
    )
    max_seq_length: int = field(
        default=120,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated. Default to the max input length of the model."
            )
        },
    )
    output_file: Optional[str] = field(
        default=None, metadata={"help": "Filename for storing predictions"}
    )
    
    