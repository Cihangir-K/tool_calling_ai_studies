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

import json


# HEADER = '\033[95m'
# MAVI = '\033[94m'
# YESIL = '\033[92m'
# SARI = '\033[93m'
# KIRMIZI = '\033[91m'
# BEYAZ = '\033[0m'
# BOLD = '\033[1m'
# UNDERLINE = '\033[4m'

wake_word = 'computer'
# wake_word = 'robot'
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
ses_turu = int(2) #türkçe dil için 1 ingilizce için 0erkek ve 2bayan
voice.setProperty('voice', voices[ses_turu].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan




# Define Ollama model
model_name = "dolphin-llama3"  # Replace with your desired Ollama model ollama run dolphin-llama3
# model_name = "phi3"    #ollama run phi3
# model_name = "phi3:instruct"
# model_name ="cas/minicpm-3b-openhermes-2.5-v2:latest" #ollama run cas/minicpm-3b-openhermes-2.5-v2:latest


def remove_prior_words(sentence, wake_word):
    # Verilen cümleyi kelimelere ayır
    words = ((str(sentence).lower()).split())
    
    print("words: ",words)
    print("wake_word:",wake_word+",")
    # wake_word'un indexini bul
    if wake_word+"," in words:

        try:
            index = words.index(wake_word+",")
        
        except ValueError:
            # wake_word cümlede bulunamadıysa, orijinal cümleyi döndür
            print("skınıtı no1 aga")
            return sentence
        # wake_word'den önceki kelimeleri sil
        del words[:index+1]
        
        # Kelimeleri tekrar birleştirerek yeni cümleyi oluştur
        new_sentence = ' '.join(words)
        print("new_sentence: ",new_sentence)        
        return str(new_sentence)
    
    elif wake_word in words:

        try:
            index = words.index(wake_word)
        
        except ValueError:
            # wake_word cümlede bulunamadıysa, orijinal cümleyi döndür
            print("skınıtı no2 aga")
            return sentence   

        # wake_word'den önceki kelimeleri sil
        del words[:index+1]
        
        # Kelimeleri tekrar birleştirerek yeni cümleyi oluştur
        new_sentence = ' '.join(words)
        print("new_sentence: ",new_sentence)
        return str(new_sentence)

# # Örnek kullanım
# sentence = "Hello computer, how are you doing today?"
# wake_word = "computer"
# new_sentence = remove_prior_words(sentence, wake_word)
# print(new_sentence)

# Function to take notes
def take_a_note(text):
    import time

    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    with open("C:/Users/E.C.E/anaconda3/envs/tool_calling_ai/ck/Notes.txt", "a") as f:
        f.write(time_string+"\n" + "\n" + text + "\n" + "\n")
        print("Note taken!")



def search_duck(text):
    search = DuckDuckGoSearchRun()
    result =search.run(text)
    print("\n"+"DUCK DUCK results: ",result+ "\n")
    return result

def wikipedia(text):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    print("\n"+"WIKI Result :",wikipedia.run(text)+"\n")
    return wikipedia.run(text)



template = """You are a helpful assistant and having a conversation with a human.
If human wants you to search in wikipedia, your aim is to extract words to make search in wikipedia. Since it is directly used in search engine, except the Search query: do NOT say anything else. 
If human wants you to search in internet, your aim is to find out that what human want to search with this sentence?  Formulate a standalone simple answer. Do NOT answer the question.
If human wants to take a note, generate a str.

{chat_history}
Human: {human_input}
Chatbot:"""

# If human says "search in internet" generate directly what human says.
# If human says "wikipedia search" generate a simple response for perfom a wikipedia search.

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm = ChatOllama(model=model_name,temperature=0)
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)
what_do_you_want=input("do you want speach to speach: ",)

if 'yes' in what_do_you_want :

#with stt and tts
    while True:

        # Mikrofondan ses al
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source,duration=1.5)  # Calibrate the recognizer
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

            
            #find a way to remove till computer 


            only_user_input=remove_prior_words(user_input, wake_word)

            print("only_user_input :",only_user_input)

            # Check for commands (remove search functionality)
            if "search in internet" in user_input.lower() or "search this in internet" in user_input.lower():
                # print("search in internet geldi.")
                try:
                    # search_text =llm_chain.predict(human_input="what user want to search in this sentence?  Formulate a standalone simple answer. Do NOT answer the Human input. Sentence is: "+user_input+ " Do NOT sanything else.")
                    # search_text =llm_chain.predict(human_input="Extract simple sentence from "+user_input+ " for a search engine Search query. Since it will be used to directly send to search engine, do NOT sanything else.")
                    search_text =llm_chain.predict(human_input="Extract simple sentence from "+only_user_input+ " for a search engine Search query. Reply in format of dictionary that contains Search Query")

                    print("\n"+"Search_text: ",search_text + "\n")
                    search_text_js = json.loads(search_text)
                    print("\n"+"Search_query: ",search_text_js.get("Search Query") + "\n")

                    search_response =search_duck(search_text_js.get("Search Query"))
                    summarized_response = llm_chain.predict(human_input=f"""summarize the :"{search_response}" in one sentence""")

                    print('\033[4m'+"search_response: ",search_response+ "\n")
                    print('\033[0m'+"summarized_response: ",summarized_response+ "\n")

                    voice.setProperty('voice', voices[ses_turu].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
                    voice.say(summarized_response)
                    voice.setProperty('rate', 145)  # speed of reading 145 
                    voice.runAndWait()
                    search_text_js=""

                except Exception as e:
                    print("Error:", e)

            elif "search in wikipedia" in user_input.lower():
                # print("search in wiki geldi.")
                try:
                    wiki_text =llm_chain.predict(human_input="Extract simple sentence from "+only_user_input+ " for a search engine Search query. Reply in format of dictionary that contains Search Query")
                    
                    print("\n"+"Search_text: ",wiki_text + "\n")
                    wiki_text_js = json.loads(wiki_text)
                    print("\n"+"Search_query: ",wiki_text_js.get("Search Query") + "\n")

                    Response_of_wiki_Search=wikipedia(wiki_text_js.get("Search Query"))

                    summarized_response=llm_chain.predict(human_input="summarize the " +Response_of_wiki_Search+ " in one sentence") 

                    print('\033[0m'+"summarized_response: ",summarized_response+ "\n")
                    voice.setProperty('voice', voices[ses_turu].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
                    voice.say(summarized_response)
                    voice.setProperty('rate', 145)  # speed of reading 145 
                    voice.runAndWait()
                    Response_of_wiki_Search=""
                    
                except Exception as e:
                    print("Error:", e)

                
            elif "take a note" or "take a note." in only_user_input:
                # print("take a note geldi.")
                try:
                    # note_text = input("What do you want to note? ")

                    response_for_note = llm_chain.predict(human_input=only_user_input)
                    
                    # response_for_note= response_text
                    print(f"Bot: {response_for_note}")
                    text=response_for_note
                    take_a_note(text)

                    voice.setProperty('voice', voices[ses_turu].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
                    voice.say("Note taken")
                    voice.setProperty('rate', 145)  # speed of reading 145 
                    voice.runAndWait()
                except Exception as e:
                    print("Error:", e)
            
            else:
                # print("düz predict geldi.")
                try:
                    response = llm_chain.predict(human_input=only_user_input)


                    # Extract text content from AIMessage object
                    response_text = response
                    print('\033[91m'+f"Bot: {response_text}")
                    voice.setProperty('voice', voices[ses_turu].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
                    voice.say(response_text)
                    voice.setProperty('rate', 145)  # speed of reading 145 
                    voice.runAndWait()

                except Exception as e:
                    print("Error:", e)

else:
#with text to text
    while True:

        

        user_input = input("human: ")

        only_user_input=user_input
        kapat=1
        if kapat==1: 
            


            # Check for commands (remove search functionality)
            if "search in internet" in user_input.lower() or "search this in internet" in user_input.lower():
                try:
                    # search_text =llm_chain.predict(human_input="what user want to search in this sentence?  Formulate a standalone simple answer. Do NOT answer the Human input. Sentence is: "+user_input+ " Do NOT sanything else.")
                    # search_text =llm_chain.predict(human_input="Extract simple sentence from "+user_input+ " for a search engine Search query. Since it will be used to directly send to search engine, do NOT sanything else.")
                    search_text =llm_chain.predict(human_input="Extract simple sentence from "+user_input+ " for a search engine Search query. Reply in format of dictionary that contains Search Query")

                    print("\n"+"Search_text: ",search_text + "\n")
                    search_text_js = json.loads(search_text)
                    print("\n"+"Search_query: ",search_text_js.get("Search Query") + "\n")

                    search_response =search_duck(search_text_js.get("Search Query"))
                    summarized_response = llm_chain.predict(human_input="summarize the "+search_response+ " in one sentence")

                    print('\033[4m'+"search_response: ",search_response+ "\n")
                    print('\033[0m'+"summarized_response: ",summarized_response+ "\n")

                except Exception as e:
                    print("Error:", e)

            elif "search in wikipedia" in user_input.lower():
                try:
                    wiki_text =llm_chain.predict(human_input="Extract simple sentence from "+user_input+ " for a search engine Search query. Reply in format of dictionary that contains Search Query")
                    
                    print("\n"+"Search_text: ",wiki_text + "\n")
                    wiki_text_js = json.loads(wiki_text)
                    print("\n"+"Search_query: ",wiki_text_js.get("Search Query") + "\n")

                    Response_of_wiki_Search=wikipedia(wiki_text_js.get("Search Query"))

                    summarized_response=llm_chain.predict(human_input="summarize the " +Response_of_wiki_Search+ " in one sentence") 

                    print('\033[0m'+"summarized_response: ",summarized_response+ "\n")
                    
                except Exception as e:
                    print("Error:", e)

                
            elif "take a note" in only_user_input or "take a note." in only_user_input:
                try:
                    # note_text = input("What do you want to note? ")

                    response_for_note = llm_chain.predict(human_input=user_input)
                    
                    # response_for_note= response_text
                    print(f"Bot: {response_for_note}")
                    text=response_for_note
                    take_a_note(text)

                except Exception as e:
                    print("Error:", e)
            
            else:
                try:
                    response = llm_chain.predict(human_input=user_input)


                    # Extract text content from AIMessage object
                    response_text = response
                    print('\033[91m'+f"Bot: {response_text}")

                except Exception as e:
                    print("Error:", e)
