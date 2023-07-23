from abc import ABC, abstractmethod
from itertools import chain
from typing import List, TypedDict

from transformers import PreTrainedTokenizer


class ModelInput(TypedDict):
    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]


class InputBuilder(ABC):
    """Base class for building inputs from segments."""
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.mask = [tokenizer.mask_token_id]

    @abstractmethod
    def build_inputs(self, history: List[str], reply: str, max_length: int) -> ModelInput:
        raise NotImplementedError

    def mask_seq(self, sequence, seq_id):
        sequence[seq_id] = self.mask
        return sequence

    @classmethod
    def _combine_sequence(self, history: List[List[int]], reply: List[int], max_length: int, flipped=False) -> List[List[int]]:
        # Trim all inputs to max_length
        history = [s[:max_length] for s in history]
        reply = reply[:max_length]
        if flipped:
            return [reply] + history
        return history + [reply]


class BertInputBuilder(InputBuilder):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        InputBuilder.__init__(self, tokenizer)
        self.cls = [tokenizer.cls_token_id]
        self.sep = [tokenizer.sep_token_id]
        self.model_inputs = ["input_ids", "token_type_ids", "attention_mask"]
        self.padded_inputs = ["input_ids", "token_type_ids"]
        self.flipped = False


    def build_inputs(
        self, 
        history: List[str],
        reply: str, 
        max_length: int, 
        input_str: bool = True
    ) -> ModelInput:
        if input_str:
            history = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t)) for t in history]
            reply = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(reply))
        sequence = self._combine_sequence(history, reply, max_length, self.flipped)
        sequence = [s + self.sep for s in sequence]
        sequence[0] = self.cls + sequence[0]

        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        last_speaker = 0
        other_speaker = 1
        seq_length = len(sequence)
        instance["token_type_ids"] = [last_speaker if ((seq_length - i) % 2 == 1) else other_speaker
                                    for i, s in enumerate(sequence) for _ in s]
        return instance
