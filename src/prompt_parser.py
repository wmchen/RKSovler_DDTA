from copy import deepcopy

from diffusers import FluxPipeline


class PromptParser:
    def __init__(self, pipe: FluxPipeline, max_length: int = 512):
        self.pipe = pipe
        self.max_length = max_length
    
    def get_word_list(self, prompt: str):
        tokens = self.pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).input_ids

        word_list = []
        for i in range(tokens.shape[1]):
            token = tokens[0, i].item()
            if token == 1:
                break
            
            word = self.pipe.tokenizer_2.convert_ids_to_tokens(token)
            if word.startswith("▁"):
                word_list.append({"word": word[1:], "token_pos": [i]})
            else:
                word_list[-1]["word"] += word
                word_list[-1]["token_pos"].append(i)
        return word_list
    
    def get_word_in_brackets(self, prompt: str):
        words = {}
        word = ""
        flag = False
        num = 1
        for char in prompt:
            if not flag:
                if char == "[":
                    flag = True
            else:
                if char == "]":
                    flag = False
                    for item in self.get_word_list(word):
                        if str(num) not in words.keys():
                            words[str(num)] = []
                        words[str(num)].append(item["word"])
                    word = ""
                    num += 1
                else:
                    word += char
        return words
    
    def __call__(self, source: str, target: str):
        if source == target:
            raise ValueError("Source and target are the same.")
        
        clean_source = source.replace("[", "").replace("]", "")
        clean_target = target.replace("[", "").replace("]", "")
        clean_source_word_list = self.get_word_list(clean_source)
        clean_target_word_list = self.get_word_list(clean_target)

        unchange_mapper = []
        change_pos = []
        pad_mapper = []
        
        # parse edit info
        source_edit_words = self.get_word_in_brackets(source)
        target_edit_words = self.get_word_in_brackets(target)
        if len(list(source_edit_words.keys())) == len(list(target_edit_words.keys())):
            edit_type = "replace"
        elif len(list(source_edit_words.keys())) < len(list(target_edit_words.keys())):
            edit_type = "insert"
        else:
            edit_type = "delete"
        
        source_word_list = deepcopy(clean_source_word_list)
        target_word_list = deepcopy(clean_target_word_list)
        while len(source_word_list) > 0:
            source_word = source_word_list.pop(0)
            found_flag = False
            for i in range(len(target_word_list)):
                if source_word["word"] == target_word_list[i]["word"]:
                    target_word = target_word_list.pop(i)
                    found_flag = True
                    break
            
            if found_flag:
                for i in range(len(source_word["token_pos"])):
                    unchange_mapper.append((
                        source_word["token_pos"][i],
                        target_word["token_pos"][i],
                    ))
                    
        while len(target_word_list) > 0:
            target_word = target_word_list.pop(0)
            for i in range(len(target_word["token_pos"])):
                change_pos.append(target_word["token_pos"][i])
        
        # parse text
        source_tokens = self.pipe.tokenizer_2(
            clean_source,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).input_ids
        target_tokens = self.pipe.tokenizer_2(
            clean_target,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).input_ids
        unchange_mapper_text = []
        for item in unchange_mapper:
            s_token = self.pipe.tokenizer_2.convert_ids_to_tokens(source_tokens[0, item[0]].item())
            t_token = self.pipe.tokenizer_2.convert_ids_to_tokens(target_tokens[0, item[1]].item())
            unchange_mapper_text.append((s_token, t_token))
        
        change_pos_text = []
        for item in change_pos:
            change_pos_text.append(self.pipe.tokenizer_2.convert_ids_to_tokens(target_tokens[0, item].item()))
        
        # parse padding
        source_last_token_pos = clean_source_word_list[-1]["token_pos"][-1]
        target_last_token_pos = clean_target_word_list[-1]["token_pos"][-1]
        offset = 1
        for source_pos in range(source_last_token_pos+1, self.max_length):
            target_pos = target_last_token_pos + offset
            if target_pos >= self.max_length:
                break
            pad_mapper.append((source_pos, target_pos))
            offset += 1
        
        result = {
            "edit_type": edit_type,
            "source_prompt": clean_source,
            "target_prompt": clean_target,
            "origin": {"source": source, "target": target},
            "unchange_mapper": unchange_mapper,
            "unchange_mapper_text": unchange_mapper_text,
            "change_pos": change_pos,
            "change_pos_text": change_pos_text,
            "pad_mapper": pad_mapper,
            "source_last_token_pos": source_last_token_pos,
            "target_last_token_pos": target_last_token_pos
        }
        return result
