from torch.utils.data import Dataset

class EnZhDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        src_text = " ".join(item.get("ordered_src_doc", []))
        tgt_text = "".join(item.get("doc_translation", []))

        model_inputs = self.tokenizer(
            src_text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_text, max_length=self.max_length, truncation=True,
                padding="max_length", return_tensors="pt"
            )

        input_ids = model_inputs["input_ids"].squeeze()
        attention_mask = model_inputs["attention_mask"].squeeze()
        label_ids = labels["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }
