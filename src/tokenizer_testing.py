from transformers import AutoTokenizer
decoder_name = "ai-forever/mGPT"
decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
decoder_tokenizer.decode(x)