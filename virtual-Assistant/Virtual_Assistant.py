# Imports
import subprocess
import pyttsx3
import random
import speech_recognition as sr
import array
import os
import datetime
import wikipedia
import webbrowser
import os
import pyjokes
import datetime

# Initializing...
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
assname = ("Jarvis one point o")
def speak(audio):
    engine.say(audio)
    engine.runAndWait()
def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>= 0 and hour<12:
        speak("Good Morning!")
    elif hour>= 12 and hour<18:
        speak("Good Afternoon!")   
    else:
        speak("Good Evening!")
    speak("I am")
    speak(assname)
    speak("Your assistant!")
def username():
    speak("What should i call you?")
    uname = takeCommand()
    print("Welcome, ", uname)
    speak("Welcome")
    speak(uname)
    speak("Please enter the password to authenticate your session.")
    pin = int(input("Type the password here: "))
    if pin == 1234:
        speak("Access Granted.")
        print("Access Granted.")
    else:
            print("Goodbye...")
            speak("goodbye")
            exit()
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Listening...")
        print("I can hear you, speak now...")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        speak("Query received...")
        print("I have recognised your voice please wait for few seconds...")    
        query = r.recognize_google(audio, language ='en-in')
        print(f"User said: {query}\n")
    except Exception as e:
        print(e)
        speak("I couldn't understand that.")
        speak("Please repeat.")
        print("I couldn't understand that, please repeat...")
        return "None"
    return query
if __name__ == '__main__':
    clear = lambda: os.system('cls')
    clear()
    wishMe()
    username()
    while True:
        query = takeCommand().lower()
        if 'wikipedia' in query:
            speak('Searching Wikipedia for.')
            speak(query)
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences = 3)
            speak("According to Wikipedia")
            print(results)
            speak(results)
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif 'open youtube' in query:
            speak("Here you go to Youtube\n")
            webbrowser.open("youtube.com")
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif 'open google' in query:
            speak("Here you go to Google\n")
            webbrowser.open("google.com")
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif 'open stack overflow' in query:
            speak("Here you go to Stack Overflow.")
            webbrowser.open("stackoverflow.com")
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        # elif ' what is the time' in query:
        #     strTime = datetime.datetime.now()
        #     speak(f"The time is {strTime}")
        #     print( "if you want to continue enter yes")
        #     speak("if yo want to continue press y")
        #     key = input("enter your decision")
        #     if key in ('y','yes','YES','Y'):
        #         print('Thank You')
        #     else:
        #         break
        elif 'how are you' in query:
            speak("Very fine Sir, Thank you for asking.")
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif "change my name" in query:
            query = query.replace("change my name to", "")
            assname = query
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif "change your name" in query:
            speak("What would you like to call me, Sir ")
            assname = takeCommand()
            speak("Thanks for naming me")
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif "what's your name" in query or "What is your name" in query:
            speak("My users call me")
            speak(assname)
            print("My users call me", assname)
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif 'exit' in query:
            speak("Thanks for giving me your time")
            exit()
        elif "who made you" in query or "who created you" in query: 
            speak("I have been created by Bharath and refined by Saraj.")
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif 'say a joke' in query:
            speak(pyjokes.get_joke())
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif 'search' in query:
            query = query.replace("search", "")          
            webbrowser.open(query)
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif 'time' in query:         
            webbrowser.open('time')
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif "who am i" in query:
            speak("You are a User.")
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif "why do you exist" in query:
            speak("I am not sure why I exist")
            speak("But my creators have created me to serve my users.")
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif "who are you" in query:
            speak("I am your virtual assistant created by Bharath and refined by Saraj.")
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif 'shutdown' in query:
                speak("Hold On! Your system is on its way to shut down.")
                subprocess.call('shutdown /r')
        elif "don't listen" in query or "stop listening" in query:
            speak("I will stop listening to you until you press W.")
            print( "Press W to wake me up.")
            key = input("Wake me up: ")
            if key in ('w', 'W'):
                print('Decision Acknowledged.')
            else:
                break
        elif "where is" in query:
            query = query.replace("where is", "")
            location = query
            speak("You asked to locate")
            speak(location)
            webbrowser.open("https://www.google.com/maps/place/"+ location)
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif "open wikipedia" in query:
            webbrowser.open("wikipedia.com")
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif "how are you" in query:
            speak("I'm fine, glad you asked!")
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
        elif "password" in query:
            MAX_LEN = 12
            DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            LOCASE_CHARACTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
			                	'i', 'j', 'k', 'm', 'n', 'o', 'p', 'q',
					            'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
					            'z']

            UPCASE_CHARACTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
				                'I', 'J', 'K', 'M', 'N', 'O', 'P', 'Q',
					            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
					            'Z']
            SYMBOLS = ['@', '#', '$', '%', '=', ':', '?', '.', '/', '|', '~', '>',
		              '*', '(', ')', '<']
            COMBINED_LIST = DIGITS + UPCASE_CHARACTERS + LOCASE_CHARACTERS + SYMBOLS
            rand_digit = random.choice(DIGITS)
            rand_upper = random.choice(UPCASE_CHARACTERS)
            rand_lower = random.choice(LOCASE_CHARACTERS) 
            rand_symbol = random.choice(SYMBOLS)
            temp_pass = rand_digit + rand_upper + rand_lower + rand_symbol
            for x in range(MAX_LEN - 4):
                temp_pass = temp_pass + random.choice(COMBINED_LIST)
                temp_pass_list = array.array('u', temp_pass)
                random.shuffle(temp_pass_list)
            password = ""
            for x in temp_pass_list:
                password = password + x
            speak("Your new password has been generated.")
            print("--------------------------")
            print(password)
            print("--------------------------")
            print( "If you want to continue, enter Y")
            speak("If you want to continue, enter y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Decision Acknowledged.')
            else:
                break
       