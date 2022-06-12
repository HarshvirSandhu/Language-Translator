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
src_text = [">>"+lang_dict[lang]+"<<" + text]
if len(text) > 0:
    bert_tokens = bert_tokenizer(text, return_tensors='pt')
    sentiment = classifier.config.id2label[F.softmax(classifier(**bert_tokens).logits, dim=1).argmax().item()]
    tokens = tokenizer(src_text, return_tensors='pt', padding=True, truncation=True)
    translate = model.generate(**tokens)
    final = [tokenizer.decode(t, skip_special_tokens=True) for t in translate]
    st.write(final[0])
    st.text("Emotion: " + str(sentiment))

    play = gTTS(final[0], lang_check=True, slow=False)
    play.save("play.mp3")
    audio_file = open("play.mp3", 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, 'mp3')