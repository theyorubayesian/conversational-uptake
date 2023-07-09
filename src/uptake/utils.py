import re
from itertools import chain

from cleantext import clean
from num2words import num2words

from transformers import PreTrainedTokenizerBase


def number_to_words(num):
    try:
        return num2words(re.sub(",", "", num))
    except:
        return num


clean_str = lambda s: clean(
    s,
    fix_unicode=True,                   # fix various unicode errors
    to_ascii=True,                      # transliterate to closest ASCII representation
    lower=True,                         # lowercase text
    no_line_breaks=True,                # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                       # replace all URLs with a special token
    no_emails=True,                     # replace all email addresses with a special token
    no_phone_numbers=True,              # replace all phone numbers with a special token
    no_numbers=True,                    # replace all numbers with a special token
    no_digits=False,                    # replace all digits with a special token
    no_currency_symbols=False,          # replace all currency symbols with a special token
    no_punct=False,                     # fully remove punctuation
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number=lambda m: number_to_words(m.group()),
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"
)


clean_str_nopunct = lambda s: clean(
    s,
    fix_unicode=True,                   # fix various unicode errors
    to_ascii=True,                      # transliterate to closest ASCII representation
    lower=True,                         # lowercase text
    no_line_breaks=True,                # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                       # replace all URLs with a special token
    no_emails=True,                     # replace all email addresses with a special token
    no_phone_numbers=True,              # replace all phone numbers with a special token
    no_numbers=True,                    # replace all numbers with a special token
    no_digits=False,                    # replace all digits with a special token
    no_currency_symbols=False,          # replace all currency symbols with a special token
    no_punct=True,                      # fully remove punctuation
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number=lambda m: number_to_words(m.group()),
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"
)


class InputBuilder(object):
  """Base class for building inputs from segments."""

  def __init__(self, tokenizer: PreTrainedTokenizerBase):
      self.tokenizer = tokenizer
      self.mask = [tokenizer.mask_token_id]

  def build_inputs(self, history, reply, max_length):
      raise NotImplementedError

  def mask_seq(self, sequence, seq_id):
      sequence[seq_id] = self.mask
      return sequence

  @classmethod
  def _combine_sequence(self, history, reply, max_length, flipped=False):
      # Trim all inputs to max_length
      history = [s[:max_length] for s in history]
      reply = reply[:max_length]
      if flipped:
          return [reply] + history
      return history + [reply]


class BertInputBuilder(InputBuilder):
  """Processor for BERT inputs"""

  def __init__(self, tokenizer):
      InputBuilder.__init__(self, tokenizer)
      self.cls = [tokenizer.cls_token_id]
      self.sep = [tokenizer.sep_token_id]
      self.model_inputs = ["input_ids", "token_type_ids", "attention_mask"]
      self.padded_inputs = ["input_ids", "token_type_ids"]
      self.flipped = False


  def build_inputs(self, history, reply, max_length, input_str=True):
    """See base class."""
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
