import streamlit as st
from transformers import MarianTokenizer, MarianMTModel, BertTokenizer, BertForSequenceClassification
from gtts import gTTS
import torch.nn.functional as F
st.title("Language Translation WebApp")
st.header("Convert English Text To Different Languages")
model_name = "Helsinki-NLP/opus-mt-en-mul"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
bert_tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
classifier = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

lang = st.selectbox("Choose Language", ['Hindi', 'Kannada', 'Malayalam', 'Punjabi', 'French'])
lang_dict = {'Hindi': 'hin', 'Kannada': 'kan', 'Malayalam': 'mal', 'Punjabi': 'pan_Guru', 'French': 'fra'}
text = st.text_area("Enter English Text")

# src_text = [">>"+lang_dict[lang]+"<<" + text]
final = ""
if len(text) > 0:
    bert_tokens = bert_tokenizer(text, return_tensors='pt')
    sentiment = classifier.config.id2label[F.softmax(classifier(**bert_tokens).logits, dim=1).argmax().item()]

    if "." in text:
        text = text.split(sep='.')
        for i in range(len(text)):
            src_text = [">>" + lang_dict[lang] + "<<" + text[i]]
            tokens = tokenizer(src_text, return_tensors='pt', padding=True, truncation=True)
            translate = model.generate(**tokens)
            output = [tokenizer.decode(t, skip_special_tokens=True) for t in translate]
            final += output[0] + "|"

    else:
        src_text = [">>" + lang_dict[lang] + "<<" + text]
        tokens = tokenizer(src_text, return_tensors='pt', padding=True, truncation=True)
        translate = model.generate(**tokens)
        output = [tokenizer.decode(t, skip_special_tokens=True) for t in translate]
        final += output[0] + " |"

    st.write(final)
    st.text("Emotion: " + str(sentiment))

    play = gTTS(final, lang_check=True, slow=False)
    play.save("play.mp3")
    audio_file = open("play.mp3", 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, 'mp3')
