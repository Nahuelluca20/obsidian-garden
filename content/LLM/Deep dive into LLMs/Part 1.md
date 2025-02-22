---
title: 'The Power of Static Typing in Programming'
publishedAt: '2024-04-07'
summary: 'In the ever-evolving landscape of software development, the debate between dynamic and static typing continues to be a hot topic.'
---

# Introduction
---
These are my note about the video [Deep dive into LLMs like ChatGPT by Andrej Karpathy](https://www.youtube.com/watch?v=7xTGNNLPyMI&t=1430s&ab_channel=AndrejKarpathy). Here I what I learn and maybe a few lines of codes with examples.
# Steps
---
Below I list the stages of the creation an LLM.

## Pre-Training

The pre-training stage is the initial phase in creating a Large Language Model (LLM), focused on providing the model with a broad understanding of language and the world. It involves several key steps.

### **Data Acquisition and Data Processing**

At this stage, data is obtained and processed to remove sensitive information, language filtering, malware or information that is not useful.

In the video [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) is used for extract a part of data. According to Andrej all of AI labs like OpenAI, Anthropic, etc have a something like this web for train their models.
![[Pasted image 20250221093658.png]]
### Tokenization

The processed text is converted into a sequence of tokens, which serve as the model's basic units of understanding. Methods like **Byte Pair Encoding (BPE)** group common byte sequences into single tokens. GPT-4 uses 100,277 tokens.

For understand the concept see this example, I want to take the first 10 rows of the FineWeb database and convert it in to tokens. So the fist step is take the data and convert in binary.

```python
from datasets import load_dataset
import itertools

def get_text():
    ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

    first_ten_rows = list(itertools.islice(ds, 20))
    raw_text = "".join(registry["text"] for registry in first_ten_rows)

    with open("raw_data.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)

    print("20 Rows has been saved")

get_text()
```

```python
def text_to_binary(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()

        binary_data = ' '.join(format(ord(char), '08b') for char in text)

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(binary_data)

        print(f"Conversion completed. File saved in {output_file}")
    except Exception as e:
        print(f"Error: {e}")

# Uso
input_file = "raw_data.txt"
output_file = "binary.txt"

text_to_binary(input_file, output_file)
```

I get something like this:

```python
01001000 01101111 01110111 00100000 01000001 01010000 00100000 01110010
1100101 01110000 01101111 01110010 01110100 01100101 01100100 00100000
01101001 01101110 00100000 01100001 01101100 01101100 00100000 01100110
01101111 01110010 01101101 01100001 01110100 01110011 00100000 01100110
01110010 01101111 01101101 00100000 01110100 01101111 01110010 01101110
01100001 01100100 01101111 00101101 01110011 01110100 01110010 01101001
01100011 01101011 01100101 01101110 00100000 01110010 01100101 01100111 ....
```

Each of this bits can be represented like bytes in a range of 0 to 255:

```python
def binary_file_to_bytes_text(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            binary_data = file.read().strip()

        bytes_list = binary_data.split()

        byte_values = [str(int(byte, 2)) for byte in bytes_list]

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(' '.join(byte_values))

        print(f"Conversion completed. File saved in {output_file}")
    except Exception as e:
        print(f"Error: {e}")

input_file = "binary.txt"
output_file = "bytes_output.txt"
binary_file_to_bytes_text(input_file, output_file)
```

Then we have a list of number where each of these numbers is in the rage between 0 to 255

```python
72 111 119 32 65 80 32 114 101 112 111 114 116 101 100 32 105 110 32
97 108 108 32 102 111 114 109 97 116 115 32 102 114 111 109 32 116 111
114 110 97 100 111 45 115 116 114 105 99 107 101 110 32 114 101 103 105
111 110 115 77 97 114 99 104 32 56 44 ....
```

Now I can group byte sequences that are repeated. For example `32 119` which is repeated `270` times in a new symbol with a new id. This is called `byte pair encoding` and I can repeat several times.

min 23 for now
