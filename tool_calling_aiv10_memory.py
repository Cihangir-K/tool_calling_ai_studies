from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from faster_whisper import WhisperModel
import speech_recognition as sr
import pyttsx3
import os
import time


wake_word = 'jarvis'
# wake_word = 'pudding'

listening_for_wake_word = True

whisper_size ='tiny'
num_cores = os.cpu_count()
whisper_model = WhisperModel(
    whisper_size,
    device = 'cpu',
    compute_type='int8',
    cpu_threads=num_cores,
    num_workers=num_cores

)
def wav_to_text(audio_path):
    segments, _ =whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

# Recognizer
recognizer = sr.Recognizer()

#(text-to-speech) 
voice = pyttsx3.init('sapi5')

voices = voice.getProperty('voices') # sesleri almak için 
voice.setProperty('voice', voices[0].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan




# Define Ollama model
model_name = "dolphin-llama3"  # Replace with your desired Ollama model
# model_name = "phi3"  
# model_name ="cas/minicpm-3b-openhermes-2.5-v2:latest"


# Function to take notes
def take_a_note(text):
    import time

    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    with open("C:/Users/E.C.E/anaconda3/envs/tool_calling_ai/ck/Notes.txt", "a") as f:
        f.write(time_string+"\n" + "\n" + text + "\n" + "\n")
        print("Note taken!")

        voice.setProperty('voice', voices[0].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
        voice.say("Note taken")
        voice.setProperty('rate', 145)  # speed of reading 145 
        voice.runAndWait()



template = """You are a chatbot named jarvis and having a conversation with a human. and you can take a note about what human says. please response only what asked from you to take note.

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm = ChatOllama(model=model_name)
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)


while True:

    # Mikrofondan ses al
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Calibrate the recognizer
        print("Listened for ambient noise ...")
        # beep()
        print('\033[91m'+"Dinliyorum...")
        audio_data = recognizer.listen(source)

    wake_audio_path = 'wake_detect.wav'
    with open(wake_audio_path, 'wb') as f:
        f.write(audio_data.get_wav_data())
        text_input = wav_to_text(wake_audio_path)
        print('text_input: ',text_input)

    user_input =text_input  
    ai_name = wake_word


    kapat=1
    if ai_name in user_input.lower(): 
        
        response = llm_chain.predict(human_input=user_input)


        # Extract text content from AIMessage object
        response_text = response
        print(f"Bot: {response_text}")
        voice.setProperty('voice', voices[0].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
        voice.say(response_text)
        voice.setProperty('rate', 145)  # speed of reading 145 
        voice.runAndWait()

        # Check for commands (remove search functionality)
        if "search" in user_input.lower():
            print("no search available for now")
            # Removed search logic

        elif "take note" in user_input.lower():

            # note_text = input("What do you want to note? ")

            # response_for_note = llm_chain.predict(human_input="summarize the "+note_text)
            
            response_for_note= response_text
            print(f"Bot: {response_for_note}")
            text=response_for_note
            take_a_note(text)
        
