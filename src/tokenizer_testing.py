from transformers import AutoTokenizer
decoder_tokenizer = AutoTokenizer.from_pretrained("ydshieh/vit-gpt2-coco-en")
decoder_tokenizer.batch_decode()
# from tokenizer_testing import decode_tokens
# decode_tokens(input, decoder_name = "ydshieh/vit-gpt2-coco-en")
def decode_tokens(input, decoder_name = "ai-forever/mGPT"):
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    if len(input.shape) > 1:
        return decoder_tokenizer.batch_decode(input)
    else:
        return decoder_tokenizer.decode(input)