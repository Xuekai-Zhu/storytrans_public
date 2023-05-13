from transformers import MarianMTModel, MarianTokenizer
src_text = [
    '>>fra<< this is a sentence in english that we want to translate to french',
    '>>por<< This should go to portuguese',
    '>>esp<< And this to Spanish'
]
torch_device = 'cpu'
model_name = 'Helsinki-NLP/opus-mt-en-zh'
tokenizer = MarianTokenizer.from_pretrained(model_name)
#print(tokenizer.supported_language_codes)

model = MarianMTModel.from_pretrained(model_name).to(torch_device)
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True).to(torch_device))
print([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
