import subprocess
import wolframalpha
import pyttsx3
import tkinter as tk
import json
import random
import operator
import speech_recognition as sr
import pandas as pd
import mediapipe as mp
import cv2
import array
import phonenumbers
from phonenumbers import geocoder,carrier
import smtplib
from audioop import add
from google.protobuf.json_format import MessageToDict
from logging import exception
from re import sub
import os
import datetime
import wikipedia
import webbrowser
import tkinter.ttk as ttk
import tkinter.font as font_
import os
import winshell
import pyautogui
import csv
import numpy as np
from PIL import Image,ImageTk
import pyjokes
import feedparser
from pathlib import Path
import smtplib
import ctypes
import time
import datetime
from tkinter import Message,Text
import requests
import shutil
from twilio.rest import Client
from clint.textui import progress
from ecapture import ecapture as ec
from bs4 import BeautifulSoup
import win32com.client as wincl
from urllib.request import urlopen
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
        speak("Good Morning Sir")
    elif hour>= 12 and hour<18:
        speak("Good Afternoon")   
    else:
        speak("Good Evening")  
    speak("I am your Assistant")
    speak(assname)
def username():
    speak("What should i call you")
    uname = takeCommand()
    speak("Welcome")
    speak(uname)
    columns = shutil.get_terminal_size().columns
    print("Welcome, ", uname.center(columns))
    speak("Please enter the password to authenticate your session.")
    pin = int(input("Type the password here:"))
    if pin == 1234:
        speak("Access Granted.")
        print("Access Granted.")
    else:
        while restart not in pin:
            print("goodbye")
            break
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        speak("I can hear you, Speak now.")
        print("I can hear you speak now.")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        speak("Query is being processed.")
        print("I have recognised your voice please wait for few seconds")    
        query = r.recognize_google(audio, language ='en-in')
        print(f"User said: {query}\n")
    except Exception as e:
        print(e)
        speak("I couldn't understand that, please repeat.")
        print("I couldn't understand that, please repeat.")  
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
            speak('Searching Wikipedia...')
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences = 3)
            speak("According to Wikipedia")
            print(results)
            speak(results)
            speak ("if you want to continue the process press y")
            print("if you want to continue the process press y")
            if option == 1:
                print( "if you want to continue enter yes")
                option = int(input('What Would you like to choose?'))
                if key in ('y','Y'):
                    print('Thank You')
                else:
                    break 
        elif 'open youtube' in query:
            speak("Here you go to Youtube\n")
            webbrowser.open("youtube.com")
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
        elif 'open google' in query:
            speak("Here you go to Google\n")
            webbrowser.open("google.com")
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
        elif 'open stack overflow' in query:
            speak("Here you go to Stack Overflow.")
            webbrowser.open("stackoverflow.com")
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
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
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
        elif "change my name" in query:
            query = query.replace("change my name to", "")
            assname = query
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
        elif "change your name" in query:
            speak("What would you like to call me, Sir ")
            assname = takeCommand()
            speak("Thanks for naming me")
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                speak('Thank You')
            else:
                break
        elif "what's your name" in query or "What is your name" in query:
            speak("My users call me")
            speak(assname)
            print("My users call me", assname)
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
        elif 'exit' in query:
            speak("Thanks for giving me your time")
            exit()
        elif "who made you" in query or "who created you" in query: 
            speak("I have been created by Bharath.")
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
        elif 'say a joke' in query:
            speak(pyjokes.get_joke())
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
        elif 'search' in query:
            query = query.replace("search", "")          
            webbrowser.open(query)
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break 
        elif "who am i" in query:
            speak("You are a User.")
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
        elif "why do you exist" in query:
            speak("I am not sure why I exist")
            speak("But my creators have created me to serve my users.")
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision: ")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
        elif "who are you" in query:
            speak("I am your virtual assistant created by bharath")
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
        elif 'reason for you' in query:
            speak("I was created as a Minor project by Mister bharath ") 
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
        elif 'news' in query:
            try: 
                jsonObj = urlopen('''https://newsapi.org / v1 / articles?source = the-times-of-india&sortBy = top&apiKey =\\times of India Api key\\''')
                data = json.load(jsonObj)
                i = 1
                speak('here are some news from the times of india')
                print('''=============== TIMES OF INDIA ============'''+ '\n')
                for item in data['articles']:
                    print(str(i) + '. ' + item['title'] + '\n')
                    print(item['description'] + '\n')
                    speak(str(i) + '. ' + item['title'] + '\n')
                    i += 1
            except Exception as e:
                print(str(e))
                print( "if you want to continue enter yes")
                speak("if yo want to continue press y")
                key = input("enter your decision")
                if key in ('y','yes','YES','Y'):
                    print('Thank You')
                else:
                    break
        elif 'shutdown' in query:
                speak("Hold On a Sec ! Your system is on its way to shut down")
                subprocess.call('shutdown / p /f')
                print( "if you want to continue enter yes")
                speak("if yo want to continue press y")
                key = input("enter your decision")
                if key in ('y','yes','YES','Y'):
                    print('Thank You')
                else:
                    break
        elif "don't listen" in query or "stop listening" in query:
            speak("for how much time you want to stop jarvis from listening commands")
            a = int(takeCommand())
            time.sleep(a)
            print(a)
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
        elif "where is" in query:
            query = query.replace("where is", "")
            location = query
            speak("User asked to Locate")
            speak(location)
            webbrowser.open("https://www.google.nl / maps / place/" + location + "")
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
        elif "restart" in query:
            subprocess.call(["shutdown", "/r"])
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break

        # elif "write a note" in query:
        #     speak("What should i write, sir")
        #     note = takeCommand()
        #     file = open('jarvis.txt', 'w')
        #     speak("Sir, Should i include date and time")
        #     snfm = takeCommand()
        #     print( "if you want to continue enter yes")
        #     speak("if yo want to continue press y")
        #     key = input("enter your decision")
        #     if key in ('y','yes','YES','Y'):
        #         print('Thank You')
        #     else:
        #         break
        #     if 'yes' in snfm or 'sure' in snfm:
        #         strTime = datetime.datetime.now().strftime("% H:% M:% S")
        #         file.write(strTime)
        #         file.write(" :- ")
        #         file.write(note)
        #     else:
        #         file.write(note)
        #         print( "if you want to continue enter yes")
        #         speak("if yo want to continue press y")
        #         key = input("enter your decision")
        #         if key in ('y','yes','YES','Y'):
        #             print('Thank You')
        #         else:
        #             break
        # elif "show note" in query:
        #     speak("Showing Notes")
        #     file = open("jarvis.txt", "r") 
        #     print(file.read())
        #     speak(file.read(6))
        #     print( "if you want to continue enter yes")
        #     speak("if yo want to continue press y")
        #     key = input("enter your decision")
        #     if key in ('y','yes','YES','Y'):
        #         print('Thank You')
        #     else:
        #         break
        elif "weather" in query:
            api_key = "Api key"
            base_url = "http://api.openweathermap.org / data / 2.5 / weather?"
            speak(" City name ")
            print("City name : ")
            city_name = takeCommand()
            complete_url = base_url + "appid =" + api_key + "&q =" + city_name
            response = requests.get(complete_url) 
            x = response.json() 
            if x["code"] != "404": 
                y = x["main"] 
                current_temperature = y["temp"] 
                current_pressure = y["pressure"] 
                current_humidiy = y["humidity"] 
                z = x["weather"] 
                weather_description = z[0]["description"] 
                speak("\n description = " +str(weather_description))
            else: 
                speak(" City Not Found ")
                print( "if you want to continue enter yes")
                speak("if you want to continue, press y")
                key = input("enter your decision: ")
                if key in ('y','yes','YES','Y'):
                    print('Thank You')
                else:
                    break
        elif "open wikipedia" in query:
            webbrowser.open("wikipedia.com")
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
        elif "how are you" in query:
            speak("I'm fine, glad you asked!")
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
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
            print(password)
            print( "if you want to continue enter yes")
            speak("if yo want to continue press y")
            key = input("enter your decision")
            if key in ('y','yes','YES','Y'):
                print('Thank You')
            else:
                break
       