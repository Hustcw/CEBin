from transformers import RobertaTokenizerFast
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
        seq_len = 0
        for key, value in function.items():
            if len(value) > 20:
                continue
            for i in range(len(value)):
                if value[i] in function.keys():
                    index = value[i].split("@")[0]
                    value[i] = "INSTR" + index
            tokens = self.tokenize(" ".join(value), max_length=self.max_length, truncation=True)
            len_token = len(tokens)
            instr_index = "INSTR" + key.split("@")[0]
            instructions = [instr_index] * len_token
            tokenized_functions["token"].extend(tokens)
            tokenized_functions["instr"].extend(instructions)

            seq_len += len_token
            if seq_len > self.max_length:
                tokenized_functions["token"] = tokenized_functions["token"][:self.max_length]
                tokenized_functions["instr"] = tokenized_functions["instr"][:self.max_length]
                break
        return tokenized_functions

    def encode_function(self, function):
        tokenized_functions = self.tokenize_function(function)
        if len(tokenized_functions["token"]) == 0:
            return None
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
