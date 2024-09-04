import os
import json
import torch
import random
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, normalizers

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def BPE(texts, vocab_size, file_path, logger):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
    )

    tokenizer.train_from_iterator(texts, trainer)
    folder = "/".join(file_path.split("/")[:-1])
    tokenizer_path = os.path.join(
        folder, "BPE" + "_" + str(vocab_size) + ".json")
    tokenizer.save(tokenizer_path, pretty=True)
    logger.info("Creating vocabulary to file %s", tokenizer_path)

    return tokenizer


class DistilledDataset(Dataset):
    def __init__(self, args, vocab_size, file_path, logger):
        postfix = file_path.split("/")[-1].split(".")[0]
        self.examples = []
        logger.info("Creating features from file at %s ", file_path)

        folder = "/".join(file_path.split("/")[:-1])

        data = []
        with open(file_path) as f:
            for line in f:
                data.append(json.loads(line.strip()))
        tokenizer_path = os.path.join(
            folder, "BPE" + "_" + str(vocab_size) + ".json")

        if os.path.exists(tokenizer_path):
            tokenizer = Tokenizer.from_file(tokenizer_path)
            logger.info("Loading vocabulary from file %s", tokenizer_path)
        else:
            texts = []
            for d in data:
                if "code_tokens" in d:
                    texts.append(" ".join(d["code_tokens"]).split())
                else:
                    texts.append(" ".join(d["function_tokens"]).split())
            tokenizer = BPE(texts, vocab_size, file_path, logger)
        
        if 'train_stud' in postfix:
            softlabs = np.load(os.path.join(folder, 'preds_unlabel_train.npy')).tolist()

        for i,d in enumerate(tqdm(data)):
            #js = json.loads(d)
            if 'train_stud' in postfix:
                softlab = softlabs[i]
            else:
                softlab = [0.1, 0.1]
            self.examples.append(convert_examples_to_features(d, tokenizer, args, softlab))
            # code = " ".join(d["func"].split())
            # source_ids = tokenizer.encode(code).ids[:args.block_size-2]
            # source_ids = [tokenizer.token_to_id(
            #     "<s>")]+source_ids+[tokenizer.token_to_id("</s>")]
            # padding_length = args.block_size - len(source_ids)
            # source_ids += [tokenizer.token_to_id("<pad>")] * padding_length
            # if "train" in postfix:
            #     self.examples.append(
            #         (InputFeatures(code, source_ids, d["pred"], d["pred"], d["soft_label"])))
            # else:
            #     self.examples.append(
            #         (InputFeatures(code, source_ids, d["target"])))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids), torch.tensor(self.examples[i].soft_label)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def convert_examples_to_features(js,tokenizer,args, soft_lab):
    #code
    if 'code_tokens' in js:
        code=' '.join(js['code_tokens'])
    else:
        code=' '.join(js['function_tokens'])
    code_ids=tokenizer.encode(code).ids[:args.block_size-2]
    code_ids =[tokenizer.token_to_id("<s>")]+code_ids+[tokenizer.token_to_id("</s>")]
    # code_ids =  tokenizer.token_to_id(code_tokens)
    padding_length = args.block_size - len(code_ids)
    code_ids+=[tokenizer.token_to_id("<pad>")]*padding_length
    
    nl=' '.join(js['docstring_tokens'])
    nl_ids=tokenizer.encode(nl).ids[:args.block_size-2]
    nl_ids =[tokenizer.token_to_id("<s>")]+nl_ids+[tokenizer.token_to_id("<s>")]
    # nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.block_size - len(nl_ids)
    nl_ids+=[tokenizer.token_to_id("<pad>")]*padding_length    
    
    return InputFeatures(code_ids,code_ids,nl_ids,nl_ids,js['url'],soft_lab,js['idx'])

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,
                 soft_pred,
                 idx,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url
        self.soft_label = soft_pred
        self.idx=idx

