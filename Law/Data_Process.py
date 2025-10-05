import json
from datasets import load_dataset
with open("DISC-Law-SFT-Pair-QA-released.jsonl" , 'r' , encoding='utf-8') as input_file , open("./LoRA_data.jsonl" , 'w', encoding='utf-8') as output_file:
    for line in input_file:
        data = json.loads(line.strip())
        LoRA_data = {
                "instruction": "你是一名专业律师，请根据中国法律回答以下问题。",
                "input": data["input"],
                "output": data["output"]
        }

        output_file.write(json.dumps(LoRA_data ,  ensure_ascii=False) + "\n")

data = load_dataset("json", data_files="LoRA_data.jsonl" , split="train")
data = data['train']
print(data[0])
