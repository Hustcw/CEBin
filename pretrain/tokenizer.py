from transformers import RobertaTokenizerFast, PreTrainedTokenizer
from transformers import BatchEncoding

class CebinTokenizer(RobertaTokenizerFast):
    # def __init__(self, tokenizer_path, max_length=512):
    #     # super(CebinTokenizer, self).__init__()
    #     self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    #     self.max_length = max_length
    #     self._bos_token = self.tokenizer._bos_token
    #     self._eos_token = self.tokenizer._eos_token
    #     self._pad_token = self.tokenizer._pad_token
    #     self._mask_token = self.tokenizer._mask_token
    #     self._pad_token_type_id = self.tokenizer._pad_token_type_id
    #     self.add_prefix_space = self.tokenizer.add_prefix_space
    #     self.deprecation_warnings = self.tokenizer.deprecation_warnings

    # @property
    # def vocab(self):
    #     return self.tokenizer.get_vocab()

    # def tokenize(self, text, max_length):
    #     return self.tokenizer.tokenize(text, max_length=max_length, truncation=True)

    # def convert_tokens_to_ids(self, tokens):
    #     return self.tokenizer.convert_tokens_to_ids(tokens)

    # def convert_ids_to_tokens(self, ids):
    #     return self.tokenizer.convert_ids_to_tokens(ids)

    # def get_vocab(self):
    #     return self.vocab

    @property
    def pad_token_type_id(self) -> int:
        """
        `int`: Id of the padding token type in the vocabulary.
        """
        return self.pad_token_id

    def tokenize_function(self, function):
        tokenized_functions = {"token": [], "instr": []}
        for key, value in function.items():
            for i in range(len(value)):
                if value[i] in function.keys():
                    index = value[i].split("@")[0]
                    value[i] = "INSTR" + index
            tokens = self.tokenize(" ".join(value), max_length=self.max_len, truncation=True)
            instr_index = "INSTR" + key.split("@")[0]
            instructions = [instr_index] * len(tokens)
            tokenized_functions["token"].extend(tokens)
            tokenized_functions["instr"].extend(instructions)
        return tokenized_functions

    def encode_function(self, function):
        tokenized_functions = self.tokenize_function(function)
        token_ids = self.convert_tokens_to_ids(tokenized_functions["token"])
        instr_ids = self.convert_tokens_to_ids(tokenized_functions["instr"])
        return BatchEncoding({
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
            "token_type_ids": instr_ids,
        })

    # def pad(self, *args, **kwargs):
    #     return super(CebinTokenizer, self).pad(*args, **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
