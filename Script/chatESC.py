# -*- coding: utf-8 -*-
"""De-Stress Chatbot.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nJOL3jGeZyfNRaxrWqK26mLz4VQd7xZo

# Funciones
"""

def es_pregunta(entrada):
  for i in entrada:
    if i == '?':
      salida = True
    else:
      salida = False
  return salida

import unicodedata
def quitar_acentos(string):
    acentos = set(map(unicodedata.lookup, ('COMBINING ACUTE ACCENT', 'COMBINING GRAVE ACCENT', 'COMBINING TILDE')))
    chars = [c for c in unicodedata.normalize('NFD', string) if c not in acentos]
    return unicodedata.normalize('NFC', ''.join(chars))

def remover_puntuacion(entrada):
  out_string = ""
  for i in entrada:
    if i not in string.punctuation:
      out_string += i
  return out_string

def preparar_texto(entrada):
  temp_string = entrada.lower()
  temp_string = remover_puntuacion(temp_string)
  temp_string = quitar_acentos(temp_string)
  lista_salida = temp_string.split()
  return lista_salida

def responder_echo(entrada, numero_bromas,espaciador):
  if entrada != None:
    echo_salida = (entrada + espaciador) * numero_bromas
  else:
    echo_salida = None
  return echo_salida

def selector(lista_entrada, checar_lista, regresar_lista):
  salida = None
  for i in lista_entrada:
    if i in checar_lista:
      salida = random.choice(regresar_lista)
      break
  return salida

def concatenar_string(string1, string2, separador):
  salida = string1 + separador + string2
  return salida

def lista_a_cadena(lista_entrada, separador):
  salida = lista_entrada[0]
  for i in lista_entrada[1:]:
    salida = concatenar_string(salida, i, separador)
  return salida

def terminar_chat(lista_entrada):
  if 'adios' in lista_entrada:
    salida = True
  else:
    salida = False
  return salida

def esta_en_lista(lista_uno, lista_dos): #Checar si cualqueir elemento esta en dos listas.

    for elemento in lista_uno:
        if elemento in lista_dos:
            return True
    return False

def encontrar_en_lista(lista_uno, lista_dos): # Find and return an element from list_one that is in list_two, or None otherwise.
    for elemento in lista_uno:
        if elemento in lista_dos:
            return elemento
    return None

def contar_puntos(entrada):
  p = 0
  h = []
  for i in entrada:
    o = i.count('.')
    if o == 1:
      p += 1
      if p == 5:
        break
    h.append(i)
  h.append('.')
  return h

"""# Librerias"""

#!pip install covid
from covid import Covid
import string
import random
import nltk
import pandas as pd
import numpy as np
import textwrap
import cv2
import speech_recognition as SRG
import time
import sys

st = SRG.Recognizer()

"""# Información"""

#!git clone https://github.com/ChatBotChallengeCdCMX/ChatBotForCovidDe-stress.git

Hombres = pd.read_csv('D:/Machine Learning Resources/Supervised/ChatBotForCovidDe-stress/BasesDeDatos/nombreshombres .csv')
Mujeres = pd.read_csv('D:/Machine Learning Resources/Supervised/ChatBotForCovidDe-stress/BasesDeDatos/nombresmujeres.csv')
Hombres = list(Hombres.iloc[:,0])
Mujeres = list(Mujeres.iloc[:,0])
Nombres = Hombres + Mujeres
Musica = pd.read_csv('D:/Machine Learning Resources/Supervised/ChatBotForCovidDe-stress/BasesDeDatos/Music.csv')
Musica = pd.DataFrame(Musica)
categorias_musica = list(pd.unique(Musica['terms']))
Videos = pd.read_csv('D:/Machine Learning Resources/Supervised/ChatBotForCovidDe-stress/BasesDeDatos/YTVideos.csv')
Videos = pd.DataFrame(Videos)
categorias_videos = list(pd.unique(Videos['category']))
Libros = pd.read_csv('D:/Machine Learning Resources/Supervised/ChatBotForCovidDe-stress/BasesDeDatos/booksdataset.csv')
Libros = pd.DataFrame(Libros)
categorias_libros = list(pd.unique(Libros['category']))
Wiki = pd.read_csv('D:/Machine Learning Resources/Supervised/ChatBotForCovidDe-stress/BasesDeDatos/WIKI.csv')
Wikis = pd.DataFrame(Wiki)
name_wikis = list(pd.unique(Wikis['Name']))
categorias_wikis = list(pd.unique(Wikis['Vertical1']))
Artic = pd.read_csv('D:/Machine Learning Resources/Supervised/ChatBotForCovidDe-stress/BasesDeDatos/ArxivDataClean.csv')
Artic = pd.DataFrame(Artic)
keywords_artic = list(pd.unique(Artic['title']))
VJ = pd.read_csv('D:/Machine Learning Resources/Supervised/ChatBotForCovidDe-stress/BasesDeDatos/VGClean.csv')
VJ = pd.DataFrame(VJ)
Gen_VG = list(pd.unique(VJ['Genre']))
Consola_VG = list(pd.unique(VJ['Platform']))
Netflix = pd.read_csv('D:/Machine Learning Resources/Supervised/ChatBotForCovidDe-stress/BasesDeDatos/netflix_titlesClean.csv')
Netflix = pd.DataFrame(Netflix)
Type_netflix = list(pd.unique(Netflix['type']))
Inv = pd.read_csv('D:/Machine Learning Resources/Supervised/ChatBotForCovidDe-stress/BasesDeDatos/InvestigadoresSNIClean.csv')
Inv = pd.DataFrame(Inv)
Area_inv = list(pd.unique(Inv['Área del Conocimiento']))


"""# Custom Chatbot"""

# This cell defines a collection of input and salida things our chatbot can say and respond to
Saludos_ini = ['Hola! Soy MMN Bot, mi especialidad es dar recomendaciones! ¿Que tal va tu día?', 'Hola! ¿Qué tál te sientes hoy?', 'Que onda, soy MMN Bot! ¿Como te llamas?']
SALUDOS = ['hello', 'hi', 'hey', 'hola', 'welcome', 'bonjour', 'greetings', 'que onda', 'holi']
SALUDOS_RESP = ["Hola, es cool hablar contigo!", 'Gusto en conocerte!',  "Hey - ¡Vamos a platicar un poco!"]

LEER_NOMBRES = Nombres
DECIR_NOMBRES = ['gusto en conocerte , My nombre es MMN Bot', 'esta bien curado tu nombre, es un gusto.', ", ese nombre mola!, es un gusto conocerte." ]

LEER_MUSICA = categorias_musica


LEER_LIBROS = ['no me gusta tanto','poco', 'no tanto', 'casi no', 'mas o menos', 'mucho', 'bastante', 'me encanta leer']

NOMBRES_LIBROS = categorias_libros

LEER_VIDEOS = ['entretenimiento', 'peliculas', 'estilo', 'comedia', 'tecnología', 'blogs', 'deportes','activismo', 'noticias', 'gaming', 'educación', 'animales', 'autos', 'viajes', 'ciencia']

NOMBRES_VIDEOS = dict(zip(LEER_VIDEOS, categorias_videos))
#NOMBRES_VIDEOS['science'] = NOMBRES_VIDEOS['tech']

#---LEER_SERIES = ['entretenimiento', 'peliculas', 'estilo', 'comedia', 'tecnología', 'blogs', 'deportes','activismo', 'noticias', 'gaming', 'educación', 'animales', 'autos', 'viajes', 'ciencia']

#---NOMBRES_SERIES = dict(zip(LEER_SERIES, categorias_videos))


LEER_CATEGORIAS = ['libros', 'libro', 'musica', 'videos', 'video', 'si', 'leer']
DIC_LIBROS = {'no me gusta tanto': 'short', 'poco': 'short', 'no tanto': 'short','casi no': 'short','mas o menos': 'medium', 'mucho': 'large',  'bastante': 'large','me encanta leer': 'large' }
LEER_COVID = ['cuarentena', 'covid', 'coronavirus', 'encerramiento', 'd 19', 'sars', 'corona']


LEER_COMPU = ['python', 'código', 'computadora', 'algoritmo', ]
DECIR_COMPU = ["Python es de lo que estoy hecho.", \
            "¿Sabías que estoy hecho con código!?", \
            "Las computadoras son mágicas", \
            "¿Crees que podría pasar el Test de Turing?"]

LEER_CIENT = ['turing', 'hopper', 'neumann', 'lovelace']
DECIR_CIENT = ['fue asombroso!', 'hizo muchas cosas importantes!', 'es alguien del que que valdria la pena saber mas :).']
NOMBRES_CIENT = {'turing': 'Alan', 'hopper': 'Grace', 'neumann': 'John von', 'lovelace': 'Ada'}

LEER_BROMAS = ['divertido', 'gracioso', 'ja', 'jaja', 'jajaja', 'xD']
DECIR_BROMAS = ['ja!', 'jajaja!', 'XD', 'lol']

LEER_NEGACIONES = ['matlab', 'java', 'C++']
DECIR_NEGACIONES = ["No, lo siento. :(, No me gustaria hablar por ahora de eso."]

NEGATIVAS = ['no', "no no", 'nop', 'nunca', "negativo", "ninguno"]

DESCONOCIDO = ['Bien.', 'Okay', 'Mm?', 'Si!', 'bien...', 'Ñam', 'Hum']
CHATEAR = ['¿Qué te gustaría hacer ahora?, puedo recomedarte algo de música, libros o algun video enretenido, ¿Cuál te gustaría?', 'Veamos, ¿Qué tipo de música te gusta?', '¿Quieres algo para relajarte?', 'Puedo buscar algo de buena música para ti,¿Qué genero te gusta?', 'Tengo algunos videos entretenidos!, escoge una categoría :D','¿Te gustan los videos? Tengo de diferentes categorías', 'además, tengo aqui algunos de mis libros favoritos, te gusta leer mucho, mas o menos, o solo un poco?',
            '¿Sobre que debería buscar?']

RESP_PREG = "Soy demasiado timido para a responder eso, jeje. De que otra cosa te gustaria una reomendación?"

def chatear():
    """función principal para tener un chat."""
    print(chr(27)+"[1;34m"+'Qué tál! Soy tu amigo MMN Bot! ¿Cuál es tu nombre?: \n')
    chat = True
    tunombre = None
    while chat:
        with SRG.Microphone() as s:
            print('Estoy escuchando...')
            entrada_audio = st.record(s, duration=7)
            sys.stdout.write("\033[F")
            try:
                texto_salida = st.recognize_google(entrada_audio,language="es")
            except:
                print("No he podido escucharte, intenta de nuevo")

        if tunombre != None:
          msg = texto_salida#input(chr(27)+"[1;30m"+str(tunombre) +': \t') \\ chr(27)+"[1;30m"+str(tunombre) +': \t' + texto_salida
          print(chr(27)+"[1;30m"+str(tunombre) +': \t' + msg)
        else: 
          msg = texto_salida#input(chr(27)+"[1;30m"+'INPUT : \t')
          print(chr(27)+"[1;30m"+'INPUT : \t' + msg)

        msg_salida = None

        n = msg.upper()
        n = n.split()

        w = msg.capitalize()
        w = w.split()

        pregunta = es_pregunta(msg)

        msg = preparar_texto(msg)

        if terminar_chat(msg):
            msg_salida = 'Adios!'
            chat = False


        if not msg_salida:

            salidas = []

            salidas.append(selector(msg, SALUDOS_RESP, SALUDOS_RESP))

            salidas.append(selector(msg, LEER_COMPU, DECIR_COMPU))

            for i in n:
              i = [i]
              if esta_en_lista(i, LEER_NOMBRES):
                  tunombre = encontrar_en_lista(i, LEER_NOMBRES)
                  salidas.append(lista_a_cadena([tunombre.capitalize(),
                                              selector(i, LEER_NOMBRES, DECIR_NOMBRES)], ' '))

            if esta_en_lista(msg, LEER_CIENT):
                name = encontrar_en_lista(msg, LEER_CIENT)
                salidas.append(lista_a_cadena([NOMBRES_CIENT[name], name.capitalize(),
                                            selector(msg, LEER_CIENT, DECIR_CIENT)], ' '))

            if esta_en_lista(msg, LEER_VIDEOS):
                name = encontrar_en_lista(msg, LEER_VIDEOS)
                ran = np.random.randint(0,len(Videos[Videos['category']==NOMBRES_VIDEOS[name]]))
                title = Videos[Videos['category']==NOMBRES_VIDEOS[name]][['title', 'video_id']]
                salidas.append(
                    'Si te gusta {} yo te recomendaría este vídeo "{}" , lo puedes ver en https://www.youtube.com/watch?v={}'.format(name,
                        title.iloc[ran][0], title.iloc[ran][1]))

            if esta_en_lista(msg, LEER_MUSICA):
                name = encontrar_en_lista(msg, LEER_MUSICA)
                ran = np.random.randint(0,len(Musica[Musica['terms']==name]))
                title = Musica[Musica['terms']==name][['release.name', 'artist.name']]
                salidas.append(
                    'Si te gusta el {} te recomiendo esta canción "{}" de {}'.format(name,
                        title.iloc[ran][0], title.iloc[ran][1]))

            if esta_en_lista(msg, LEER_COVID):
              covid = Covid()
              casos = covid.get_status_by_country_name(country_name='mexico')
              print(chr(27)+"[1;31m"+'CHATBOT: Aquí hay un poco de información del covid en tu zona: \n')
              for x in casos:
                print(chr(27)+"[1;31m"+ x, ':', casos[x])

              print(chr(27)+"[1;31m"+"CHATBOT: Puedes encontrar un mapa genial aquí: https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6")
              print(chr(27)+"[1;32m"+"CHATBOT: Pero hablemos de algo más relajante.")


            if esta_en_lista(msg, LEER_LIBROS):
              name = encontrar_en_lista(msg, LEER_LIBROS)
              ran = np.random.randint(0,len(Libros[Libros['category']==DIC_LIBROS[name]]))
              title = Libros[Libros['category']==DIC_LIBROS[name]][['title', 'authors', 'num_pages']]
              salidas.append(
                  'Este libro "{}" suena bien para ti, fue escrito por {} y tiene {} páginas.'.format(
                      title.iloc[ran][0], title.iloc[ran][1], title.iloc[ran][2]))

            if esta_en_lista(w, name_wikis):
                name = encontrar_en_lista(w, name_wikis)
                ran = np.random.randint(0,len(Wikis[Wikis['Name']==name]))
                title = Wikis[Wikis['Name']==name][['Name', 'WikiDescription','WikiUrl']]
                salidas.append(
                    'Aquí esta la definición de {} que encontré para ti: {}'.format(
                        title.iloc[ran][0], lista_a_cadena(contar_puntos(title.iloc[ran][1]), '')) +
                     ' Puedes leer más de ello en: {}'.format(
                        title.iloc[ran][2]))

            if esta_en_lista(msg, LEER_CATEGORIAS):
                name = encontrar_en_lista(msg, LEER_CATEGORIAS)
                salidas.append('Tengo estas categorías, ¿cuál te gustaría?'+'\n'+
                            '\n -LIBROS: {}'.format(str(categorias_libros)) + '\n' +
                            '\n -VIDEOS:{}'.format(str(LEER_VIDEOS)) + '\n' +
                            'Y de **MUSICA**, tengo cualquier genero que quieras.')

            salidas.append(responder_echo(selector(msg, LEER_BROMAS, DECIR_BROMAS), 3, ''))

            if esta_en_lista(msg, LEER_NEGACIONES):
                salidas.append(lista_a_cadena([selector(msg, LEER_NEGACIONES, DECIR_NEGACIONES), encontrar_en_lista(msg, LEER_NEGACIONES)], ' '))


            opciones = list(filter(None, salidas))
            if opciones:
                msg_salida = random.choice(opciones)

        if not msg_salida and pregunta:
            msg_salida = RESP_PREG


        if not msg_salida:
            msg_salida = random.choice(DESCONOCIDO)

        print(chr(27)+"[1;36m"+'CHATBOT:')

        for i in textwrap.wrap(str(msg_salida), 130):
          print(chr(27)+"[1;36m"+ i)

        if msg_salida != 'Adios!':
          print('\n' +chr(27)+"[1;36m"+'CHATBOT: \t'+random.choice(CHATEAR))
