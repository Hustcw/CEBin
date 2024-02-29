from transformers import RobertaTokenizerFast, PreTrainedTokenizer
from transformers import BatchEncoding

class CebinTokenizer(RobertaTokenizerFast):

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

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
