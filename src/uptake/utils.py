import re
import string

from cleantext import clean
from num2words import num2words

punct_chars = list((set(string.punctuation) | {'’', '‘', '–', '—', '~', '|', '“', '”', '…', "'", "`", '_'}))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))


def number_to_words(num: str) -> str:
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

def get_num_words(text):
    if not isinstance(text, str):
        print("%s is not a string" % text)
    text = replace.sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'\[.+\]', " ", text)
    return len(text.split())


def get_clean_text(text, remove_punct=False):
    if remove_punct:
        return clean_str_nopunct(text)
    return clean_str(text)
