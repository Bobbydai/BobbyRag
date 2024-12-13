# coding=utf-8
"""
    @project: maxkb
    @Author：虎
    @file： tokenizer_manage_config.py
    @date：2024/4/28 10:17
    @desc:
"""


class TokenizerManage:
    tokenizer = None

    @staticmethod
    def get_tokenizer():
        from transformers import GPT2TokenizerFast
        if TokenizerManage.tokenizer is None:
            TokenizerManage.tokenizer = GPT2TokenizerFast.from_pretrained(
                'gpt2',
                cache_dir="/opt/maxkb/model/tokenizer",
                local_files_only=True,
                resume_download=False,
                force_download=False)
        return TokenizerManage.tokenizer


# # 获取 tokenizer 实例
# tokenizer = TokenizerManage.get_tokenizer()

# # 测试 tokenizer 功能
# text = "Hello, how are you?"
# tokens = tokenizer.tokenize(text)
# print("Tokens:", tokens)

# # 编码和解码测试
# encoded_input = tokenizer.encode(text, return_tensors='pt')
# decoded_output = tokenizer.decode(encoded_input[0])
# print("Encoded Input:", encoded_input)
# print("Decoded Output:", decoded_output)
