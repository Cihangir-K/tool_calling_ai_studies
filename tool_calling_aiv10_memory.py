from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from faster_whisper import WhisperModel
import speech_recognition as sr
import pyttsx3
import os
import time

# HEADER = '\033[95m'
# MAVI = '\033[94m'
# YESIL = '\033[92m'
# SARI = '\033[93m'
# KIRMIZI = '\033[91m'
# BEYAZ = '\033[0m'
# BOLD = '\033[1m'
# UNDERLINE = '\033[4m'


wake_word = 'jarvis'
# wake_word = 'pudding'

listening_for_wake_word = True

whisper_size ='tiny.en'
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

def search_duck(text):
    search = DuckDuckGoSearchRun()
    result =search.run(text)
    print("\n"+"DUCK DUCK results: ",result+ "\n")
    return result

def wikipedia(text):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    print("\n"+"WIKI Result :",wikipedia.run(text)+"\n")
    return wikipedia.run(text)



template = """You are a chatbot named jarvis and having a conversation with a human.
If human wants you to wikipedia search, your aim is extract one or two word to make search in wikipedia. Formulate a standalone answer. Do NOT answer the question. 
If human wants you to search in internet, your aim is find out that what human want to search with this sentence?  Formulate a standalone simple answer. Do NOT answer the question.




{chat_history}
Human: {human_input}
Chatbot:"""

# If human says "search in internet" generate directly what human says.
# If human says "wikipedia search" generate a simple response for perfom a wikipedia search.

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
        print('\033[94m'+'text_input: ',text_input)

    user_input =text_input  
    ai_name = wake_word


    kapat=1
    if ai_name in user_input.lower(): 
        


        # Check for commands (remove search functionality)
        if "search in internet" in user_input.lower():

            search_text =llm_chain.predict(human_input="what user want to search in this sentence?  Formulate a standalone simple answer. Do NOT answer the question. Sentence is: "+user_input+ " Do NOT sanything else.")
            print("\n"+"Search_text: ",search_text + "\n")

            search_response =search_duck(search_text)
            summarized_response = llm_chain.predict(human_input="summarize the "+search_response+ " according to "+user_input+" in one sentence")
            print('\033[4m'+"search_response: ",search_response+ "\n")
            print('\033[0m'+"summarized_response: ",summarized_response+ "\n")

            voice.setProperty('voice', voices[0].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
            voice.say(summarized_response)
            voice.setProperty('rate', 145)  # speed of reading 145 
            voice.runAndWait()

        elif "wikipedia search" in user_input.lower():
            wiki_text =llm_chain.predict(human_input="Your aim is extract one word to make search in wikipedia. Sentence is: "+user_input+ " Do NOT sanything else.")
            
            print("wiki_text: ",wiki_text)
            Response_of_wiki_Search=wikipedia(wiki_text)
            summarized_response=llm_chain.predict(human_input="summarize " +Response_of_wiki_Search+" in one sentence.") 
            voice.setProperty('voice', voices[0].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
            voice.say(summarized_response)
            voice.setProperty('rate', 145)  # speed of reading 145 
            voice.runAndWait()
        elif "take note" in user_input.lower():

            # note_text = input("What do you want to note? ")

            response_for_note = llm_chain.predict(human_input=user_input)
            
            # response_for_note= response_text
            print(f"Bot: {response_for_note}")
            text=response_for_note
            take_a_note(text)
        
        else:

            response = llm_chain.predict(human_input=user_input)


            # Extract text content from AIMessage object
            response_text = response
            print('\033[91m'+f"Bot: {response_text}")
            voice.setProperty('voice', voices[0].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
            voice.say(response_text)
            voice.setProperty('rate', 145)  # speed of reading 145 
            voice.runAndWait()