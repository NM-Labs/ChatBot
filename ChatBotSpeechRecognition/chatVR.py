# -*- coding: utf-8 -*-
"""MMN Chatbot.ipynb

En este proyecto colaboraron:
-   Natalia Sánchez Patiño,  github: @Natalia-SP
-   Mario Rosas Otero,       github: @Mariuki
-   Mario Velázquez Vilchiz, github: @mvvazta

"""
### Librerias ###

#!pip install covid

import string
from covid import Covid
import random
import nltk
import pandas as pd
import numpy as np
import textwrap
import cv2
import speech_recognition as SRG
import time
import sys
import unicodedata
import pyttsx3
from deep_translator import GoogleTranslator
import pycountry
import plotly.express as px
import wget
import os

st = SRG.Recognizer()

"""#Bases de Datos"""
def getDatos(path="/"):
    global LEER_NOMBRES, DECIR_NOMBRES, OP_ENTRETENIMIENTO, OP_ACADEMICO, SALUDOS_IN, SALUDOS, SALUDOS_RESP, PREGUNTA_1, LEER_MUSICA, LEER_LIBROS, NOMBRES_LIBROS, LEER_VIDEOS, NOMBRES_VIDEOS, LEER_INV, LEER_PELIS, DIC_PELIS, DIC_INV, LEER_SERIES, DIC_SERIES, LEER_VJ_P, DIC_VJ_P, LEER_VJ_G, DIC_VJ_G, LEER_CATEGORIAS, LEER_COVID, LEER_COMPU, DECIR_COMPU, LEER_CIENT, NOMBRES_CIENT, DECIR_CIENT, LEER_BROMAS, DECIR_BROMAS, LEER_NEGACIONES, DECIR_NEGACIONES, NEGATIVAS, DESCONOCIDO, CHATEAR, RESP_PREG, Hombres, Mujeres, Nombres, Musica, categorias_musica, Videos, categorias_videos, Libros, categorias_libros, Wiki, Wikis, name_wikis, categorias_wikis, Artic, VJ, categorias_vj, categorias2_vj, Netflix, Netflix_p, categorias_netp, Netflix_s, categorias_nets, Type_netflix, Inv, categorias_inv, Area_inv 
    

        #!git clone https://github.com/NM-Labs/ChatBot.git
    #     path = "D:/GitHub/"

    # path = getDatos()

    Hombres = pd.read_csv(path +'ChatBot/BasesDeDatos/nombreshombres .csv')
    Mujeres = pd.read_csv(path+'ChatBot/BasesDeDatos/nombresmujeres.csv')
    Hombres = list(Hombres.iloc[:,0])
    Mujeres = list(Mujeres.iloc[:,0])
    Nombres = Hombres + Mujeres
    Musica = pd.read_csv(path +'ChatBot/BasesDeDatos/Music.csv')
    Musica = pd.DataFrame(Musica)
    categorias_musica = list(pd.unique(Musica['terms']))
    Videos = pd.read_csv(path +'ChatBot/BasesDeDatos/YTVideos.csv')
    Videos = pd.DataFrame(Videos)
    categorias_videos = list(pd.unique(Videos['category']))
    Libros = pd.read_csv(path +'ChatBot/BasesDeDatos/booksdataset.csv')
    Libros = pd.DataFrame(Libros)
    categorias_libros = list(pd.unique(Libros['category']))
    Wiki = pd.read_csv(path +'ChatBot/BasesDeDatos/WIKI.csv')
    Wikis = pd.DataFrame(Wiki)
    name_wikis = list(pd.unique(Wikis['Name']))
    categorias_wikis = list(pd.unique(Wikis['Vertical1']))
    Artic = pd.read_csv(path +'ChatBot/BasesDeDatos/ArxivDataClean.csv')
    Artic = pd.DataFrame(Artic)
    VJ = pd.read_csv(path +'ChatBot/BasesDeDatos/VGClean.csv')
    VJ = pd.DataFrame(VJ)
    categorias_vj = list(pd.unique(VJ['Genre']))
    categorias2_vj = list(pd.unique(VJ['Platform']))
    Netflix = pd.read_csv(path +'ChatBot/BasesDeDatos/netflix_titlesClean.csv')
    Netflix_p = pd.DataFrame(Netflix[Netflix['type']=='Movie'])
    categorias_netp = list(pd.unique(Netflix_p['listed_in']))
    Netflix_s = pd.DataFrame(Netflix[Netflix['type']=='TV Show'])
    categorias_nets = list(pd.unique(Netflix_s['listed_in']))
    Type_netflix = list(pd.unique(Netflix['type']))
    Inv = pd.read_csv(path +'ChatBot/BasesDeDatos/InvestigadoresSNIClean.csv')
    Inv = pd.DataFrame(Inv)
    categorias_inv = list(pd.unique(Inv['Área del Conocimiento']))
    Area_inv = list(pd.unique(Inv['Área del Conocimiento']))

    """# Listas de palabras frases y categorias"""

    OP_ENTRETENIMIENTO = ["videos", "peliculas", "series", "musica", "libros","videojuegos","juegos"]
    OP_ACADEMICO = ["articulo", "investigador", "investigadores", "articulos", "definiciones"]

    SALUDOS_IN = ['Hola! Soy MMN Bot, mi especialidad es dar recomendaciones! ¿Que tal va tu día?', 'Hola! ¿Qué tál te sientes hoy?', 'Que onda, soy MMN Bot! ¿Como te llamas?']
    SALUDOS = ['hello', 'hi', 'hey', 'hola', 'welcome', 'bonjour', 'greetings', 'que onda', 'holi']
    SALUDOS_RESP = ["Hola, es cool hablar contigo!", 'Gusto en conocerte!',  "Hey - ¡Vamos a platicar un poco!"]

    PREGUNTA_1 = ["¿Qué quisieras que te recomendara, tengo la sección de entretenimiento, académico y covid", "Muy bien, continuemos! ¿Buscas algo académico, de entretenimiento o información sobre Covid?", "Me caes bien, puedo recomendarte algo académico, algo de entretenimiento o de Covid, ¿cuál prefieres?", "Sos la ostía, tengo para vosotros algo de entretenimiento, de covid o algo académico, elige..."]


    LEER_NOMBRES = Nombres
    DECIR_NOMBRES = ['gusto en conocerte,  vamoa platicar :D', 'esta bien curado tu nombre, es un gusto.', ", ese nombre mola!, es un gusto conocerte.",'Gusto en conocerte!' , "Hey - ¡Vamos a platicar un poco!"]

    LEER_MUSICA = categorias_musica


    LEER_LIBROS = ['no','poco', 'corto', 'medianito', 'menos', 'mucho', 'bastante', 'largo', 'encanta']

    NOMBRES_LIBROS = categorias_libros

    LEER_VIDEOS = ['entretenimiento', 'peliculas', 'estilo', 'comedia', 'tecnologia', 'blogs', 'deportes','activismo', 'noticias', 'gaming', 'educacion', 'animales', 'autos', 'viajes', 'ciencia']

    NOMBRES_VIDEOS = dict(zip(LEER_VIDEOS, categorias_videos))
    NOMBRES_VIDEOS['ciencia'] = NOMBRES_VIDEOS['tecnologia']

    LEER_INV = ['fisica','matematicas','tierra','biologia','quimica', 'medicina', 'salud', 'humanidades','conducta', 'sociales', 'biotecnologia','agropecuarias','ingenierias']
    DIC_INV = {'fisica': categorias_inv[4], 'matematicas': categorias_inv[4], 'tierra': categorias_inv[4], 'biologia':categorias_inv[1], 'quimica':categorias_inv[1], 'medicina':categorias_inv[3], 'salud': categorias_inv[3], 'humanidades':categorias_inv[6], 'conducta':categorias_inv[6], 'sociales':categorias_inv[2], 'biotecnologia':categorias_inv[0], 'agropecuarias':categorias_inv[0], 'ingenierias':categorias_inv[5]}

    LEER_PELIS = ['documentales','accion','comedia','palomera','drama', 'terror', 'clasicos', 'ficcion','infantil']
    DIC_PELIS = {'documentales': [categorias_netp[x] for x in [0,9,26,27]], 'accion':[categorias_netp[x] for x in [1,2,16,18,24,32]], 'comedia':[categorias_netp[x] for x in [3,4,6,11,13,14,28,31,34]],'palomera':[categorias_netp[x] for x in [5]], 'drama':[categorias_netp[x] for x in [7,8,15,21,23,25,33]],'terror':[categorias_netp[x] for x in [10,19,20]],'clasicos':[categorias_netp[x] for x in [12]],'ficcion':[categorias_netp[x] for x in [17]],'infantil':[categorias_netp[x] for x in [22,29,30]]}

    LEER_SERIES = ['crimen','novela','infantil','documentales','clasicos', 'reality']
    DIC_SERIES = {'crimen': [categorias_nets[x] for x in [0,5,7,9,12,14,19]], 'novela':[categorias_nets[x] for x in [1,3,4,8,11,16,15]], 'infantil':[categorias_nets[x] for x in [2,18]],'documentales':[categorias_nets[x] for x in [6,17]], 'clasicos':[categorias_nets[x] for x in [10,13]],'reality':[categorias_nets[x] for x in [20,21]]}

    LEER_VJ_P = ['xbox','360','one','play','station','playstation', 'wii', 'psp', 'computadora', 'compu', 'pc']
    DIC_VJ_P = {'xbox': [categorias2_vj[x] for x in [4,13,17]], '360': [categorias2_vj[x] for x in [4,13,17]],'one': [categorias2_vj[x] for x in [4,13,17]], 'playstation':[categorias2_vj[x] for x in [5,6,10,12,16]],'play':[categorias2_vj[x] for x in [5,6,10,12,16]],'station':[categorias2_vj[x] for x in [5,6,10,12,16]],'psp':[categorias2_vj[x] for x in [5,6,10,12,16]], 'wii':[categorias2_vj[x] for x in [0,19]],'computadora':[categorias2_vj[x] for x in [14]],'compu':[categorias2_vj[x] for x in [14]],'pc':[categorias2_vj[x] for x in [14]]}
    LEER_VJ_G = ['deportes','plataforma','carreras','roles','rompecabezas','variado', 'disparos', 'simulacion', 'accion', 'peleas', 'aventura', "estrategia"]
    DIC_VJ_G = dict(zip(LEER_VJ_G, categorias_vj))

    LEER_CATEGORIAS = ['libros', 'libro', 'musica', 'videos', 'video', 'si', 'leer']
    DIC_LIBROS = {'no': 'short', 'poco': 'short', 'corto': 'short','medianito': 'medium','mediano': 'medium','menos': 'medium', 'mucho': 'large',  'bastante': 'large','encanta': 'large' ,'largo': 'large' }
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

#     return 
### Funciones ###

def es_pregunta(entrada):
  for i in entrada:
    if i == '?':
      salida = True
    else:
      salida = False
  return salida

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

def terminar_chat(lista_entrada):
  if encontrar_en_lista(lista_entrada, ["no", "adios","nelson","bye","chao","vemos","nel"]):
    salida = True
  else:
    salida = False
  return salida


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

def codigo_pais(nombre):
    try:
        return pycountry.countries.lookup(nombre).alpha_3
    except:
        return None

def creargrafica():
    wget.download("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", bar=None)
    df_confirm = pd.read_csv('time_series_covid19_confirmed_global.csv')
    df_confirm = df_confirm.drop(columns=['Province/State','Lat', 'Long'])
    df_confirm = df_confirm.groupby('Country/Region').agg('sum')
    date_list = list(df_confirm.columns)
    df_confirm['country'] = df_confirm.index
    df_confirm['iso_alpha_3'] = df_confirm['country'].apply(codigo_pais)
    df_long = pd.melt(df_confirm, id_vars=['country','iso_alpha_3'], value_vars=date_list)
    fig = px.choropleth(df_long,                            # Input Dataframe
                     locations="iso_alpha_3",           # identify country code column
                     color="value",                     # identify representing column
                     hover_name="country",              # identify hover name
                     animation_frame="variable",        # identify date column
                     projection="natural earth",        # select projection
                     color_continuous_scale = 'Turbo',  # select prefer color scale
                     range_color=[0,50000]              # select range of dataset
                     )
    os.remove("time_series_covid19_confirmed_global.csv")
    return fig.show()

def escuchar_mensaje(tunombre="INPUT", w=False):
    texto_salida_audio = None
    mensaje = None
    with SRG.Microphone() as s:
        print(chr(27)+"[1;31m"+'Estoy escuchando...')
        entrada_audio = st.record(s, duration=5)
        # sys.stdout.write("\033[F")
        try:
            print(chr(27)+"[1;31m"+"Procesando...")
            texto_salida_audio = st.recognize_google(entrada_audio,language="es")
            # print(texto_salida_audio)
            print(chr(27)+"[1;31m"+"Reconocido.")
            print(chr(27)+"[1;30m"+str(tunombre) +': \t' + str(texto_salida_audio))
            if w:
                mensaje = texto_salida_audio.split() #w se usa para Wikipedia
                mensaje = [mensaje.capitalize() for mensaje in mensaje]
            else:
                mensaje = preparar_texto(texto_salida_audio)
        except:
            print(chr(27)+"[1;31m"+"No he podido escucharte, intenta de nuevo")

            mensaje = escuchar_mensaje(tunombre)
    return mensaje

def hablar(msg_salida):
    sentencia = pyttsx3.init()

    sentencia.setProperty("rate",150)
    sentencia.setProperty("volume",.6)
    listVoices = sentencia.getProperty("voices")
    sentencia.setProperty("voice",listVoices[0].id)

    sentencia.say(msg_salida)
    sentencia.runAndWait()

### Caso 2 ###
def videos(tunombre):
    msg_salida = None
    msg_salida = random.choice(["¿Qué tipos de videos te gustarían?, tengo de:\n", "Genial!, tengo estas categorías:\n", "Muy bien, revisaré mi colección favorita de videos, podríamos empezar por: \n"])
    opciones_videos = " - Comedia\n - Tecnología\n - Películas\n - Estilo\n - Entretenimiento\n - Blogs\n - Deportes\n - Activismo\n - Noticias\n - Gaiming\n - Educación\n - Animales\n - Autos\n - Viajes\n - Ciencia"
    # msg_salida = msg_salida + opciones_videos
    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida+opciones_videos)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)
    print("\n"+chr(27)+"[1;34m"+'CHATBOT:\n'+opciones_videos)
    msg = escuchar_mensaje(tunombre)
    name = encontrar_en_lista(list(msg), LEER_VIDEOS)
    if name:
        ran = np.random.randint(0,len(Videos[Videos['category']==NOMBRES_VIDEOS[name]]))
        title = Videos[Videos['category']==NOMBRES_VIDEOS[name]][['title', 'video_id']]
        msg_salida = (
            'Si te gusta {} yo te recomendaría este vídeo "{}" , lo puedes ver en https://www.youtube.com/watch?v={}'.format(name,
                title.iloc[ran][0], title.iloc[ran][1]))
    else:
        msg_salida = "Hum... creo que no capté algo de lo que dijiste, ¿podrías repetirlo?"
        print(chr(27)+"[1;34m"+'CHATBOT:')
        hablar(msg_salida)
        for i in textwrap.wrap(str(msg_salida), 130):
          print(chr(27)+"[1;34m"+ i)
        videos(tunombre)
        msg_salida = []

    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)


    return

def peliculas(tunombre):
    msg_salida = None
    msg_salida = random.choice(["¿Qué clase de películas te gustan?, tengo\n", "Genial!, tengo estas categorías\n", "Muy bien, entretenimiento, podríamos empezar por: \n"])
    opciones_peliculas = " - documentales\n - accion\n - comedia\n - palomera\n - drama\n - terror\n - clasicos\n - ficcion\n - infantil"
    # msg_salida = msg_salida + opciones_peliculas
    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida+opciones_peliculas)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)
    print("\n"+chr(27)+"[1;34m"+'CHATBOT:\n'+opciones_peliculas)
    msg = escuchar_mensaje(tunombre)
    name = encontrar_en_lista(msg, LEER_PELIS)
    if name:
        select = random.choice(DIC_PELIS[name])
        ran = np.random.randint(0,len(Netflix_p[Netflix_p['listed_in']==select]))
        datos = Netflix_p[Netflix_p['listed_in']==select][['title', 'duration','description']].iloc[ran]
        msg_salida = (
            'Si te gusta la categoría de {} yo te recomendaría la película "{}", que dura {}, trata de: {}'.format(name,
                datos[0], GoogleTranslator(source='auto', target='es').translate(datos[1]), GoogleTranslator(source='auto', target='es').translate(datos[2])))
    else:
        msg_salida = "Hum... creo que no capté algo de lo que dijiste, ¿podrías repetirlo?"
        print(chr(27)+"[1;34m"+'CHATBOT:')
        hablar(msg_salida)
        for i in textwrap.wrap(str(msg_salida), 130):
          print(chr(27)+"[1;34m"+ i)
        peliculas(tunombre)
        msg_salida = []

    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)


    return

def series(tunombre):
    msg_salida = None
    msg_salida = random.choice(["¿Que tipo de series te gustan?, tengo\n", "Genial!, tengo estas categorías\n", "Muy bien, series, podríamos empezar por: \n"])
    categorias_series = " - Crimen\n - Novela\n - Infantil\n - Documentales\n - Clásicos\n - Reality Shows"
    # msg_salida = msg_salida + categorias_entretenimiento
    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida+" - Crimen - Novela - Infantil - Documentales - Clásicos - Reality Schows")
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)
    print("\n"+chr(27)+"[1;34m"+'CHATBOT:\n'+categorias_series)
    msg = escuchar_mensaje(tunombre)
    name = encontrar_en_lista(msg, LEER_SERIES)
    if name:
        select = random.choice(DIC_SERIES[name])
        ran = np.random.randint(0,len(Netflix_s[Netflix_s['listed_in']==select]))
        datos = Netflix_s[Netflix_s['listed_in']==select][['title', 'duration','description']].iloc[ran]
        msg_salida = (
    'Si te gusta la categoría de {} yo te recomendaría la serie "{}", que tiene {}, trata de: {}'.format(name,
        datos[0], GoogleTranslator(source='auto', target='es').translate(datos[1]), GoogleTranslator(source='auto', target='es').translate(datos[2])))
    else:
        msg_salida = "Hum... creo que no capté algo de lo que dijiste, ¿podrías repetirlo?"
        print(chr(27)+"[1;34m"+'CHATBOT:')
        hablar(msg_salida)
        for i in textwrap.wrap(str(msg_salida), 130):
          print(chr(27)+"[1;34m"+ i)
        series(tunombre)
        msg_salida = []

    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)


    return

def musica(tunombre):
    msg_salida = None
    msg_salida = random.choice(["¿Qué genero de música prefieres?\n", "Genial!, ¿Qué música te gusta?\n", "La música es genial!, podríamos empezar por decirme tu genero favorito \n"])

    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)

    msg = escuchar_mensaje(tunombre)
    name = encontrar_en_lista(msg, LEER_MUSICA)
    if name:
        ran = np.random.randint(0,len(Musica[Musica['terms']==name]))
        title = Musica[Musica['terms']==name][['release.name', 'artist.name']]
        msg_salida = (
            'Si te gusta el {} te recomiendo esta canción "{}" de {}'.format(name,
                title.iloc[ran][0], title.iloc[ran][1]))
    else:
        msg_salida = "Hum... creo que no capté algo de lo que dijiste, ¿podrías repetirlo?"
        print(chr(27)+"[1;34m"+'CHATBOT:')
        hablar(msg_salida)
        for i in textwrap.wrap(str(msg_salida), 130):
          print(chr(27)+"[1;34m"+ i)
        musica(tunombre)
        msg_salida = []

    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)


    return

def libros(tunombre):
    msg_salida = None
    msg_salida = random.choice(["¿Qué tanto te gusta leer?, ¿mucho, poco?\n", "Genial!, podría sugerirte un libro corto, medianito o algo largo.\n", "Muy bien, libros, ¿qué tan grandes? mucho, mas o menos, poco...: \n"])

    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)

    msg = escuchar_mensaje(tunombre)
    name = encontrar_en_lista(msg, LEER_LIBROS)
    if name:
        ran = np.random.randint(0,len(Libros[Libros['category']==DIC_LIBROS[name]]))
        title = Libros[Libros['category']==DIC_LIBROS[name]][['title', 'authors', 'num_pages']]
        msg_salida = (
         'Este libro "{}" suena bien para ti, fue escrito por {} y tiene {} páginas.'.format(
             title.iloc[ran][0], title.iloc[ran][1], title.iloc[ran][2]))
    else:
        msg_salida = "Hum... creo que no capté algo de lo que dijiste, ¿podrías repetirlo?"
        print(chr(27)+"[1;34m"+'CHATBOT:')
        hablar(msg_salida)
        for i in textwrap.wrap(str(msg_salida), 130):
          print(chr(27)+"[1;34m"+ i)
        libros(tunombre)
        msg_salida = []

    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)


    return

def juegos(tunombre):
    msg_salida = None
    msg_salida = random.choice(["¿Qué clase de videojuegos te gustan?, ¿qué plataforma usas?\n", "Genial!, dime una categoría y plataforma.\n", "Muy bien, videojuegos, ¿de qué tipo, qué consola?: \n"])
    categorias_videojuegos = " Categorías:\t Consolas:\n - Acción\t + Xbox\n - Aventuras\t + PlayStation\n - Carreras\t + Wii\n - Deportes\t + Computadora\n - Disparos\n - Estrategia\n - Peleas\n - Plataforma\n - Roles\n - Rompecabezas\n - Simulacion\n - Variado"
    # msg_salida = msg_salida + categorias_videojuegos
    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida+" Las consolas disponibles son: Xbox PlayeStation  Wii  y Computadora\n con las siguientes categorías")
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)
    print("\n"+chr(27)+"[1;34m"+'CHATBOT:\n'+categorias_videojuegos)
    msg = escuchar_mensaje(tunombre)
    keyp = encontrar_en_lista(msg, LEER_VJ_P)
    keyg = encontrar_en_lista(msg, LEER_VJ_G)
    if keyp and keyg:
        Keyp = random.choice(DIC_VJ_P[keyp])
        Keyg = DIC_VJ_G[keyg]
        VJG = VJ[VJ['Genre'] == Keyg]
        VJG = VJG[VJG['Platform'] == Keyp]
        ran = np.random.randint(0,len(VJG))
        datos = VJG[['Name','Genre', 'Platform']].iloc[ran]
        msg_salida = (
            'Si te gusta la categoría de {} yo te recomendaría "{}", para {}.'.format(keyg,
                datos[0], datos[2]))
    else:
        msg_salida = "Hum... creo que no capté algo de lo que dijiste, ¿podrías repetirlo?"
        print(chr(27)+"[1;34m"+'CHATBOT:')
        hablar(msg_salida)
        for i in textwrap.wrap(str(msg_salida), 130):
          print(chr(27)+"[1;34m"+ i)
        juegos(tunombre)
        msg_salida = []

    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)


    return

def articulos(tunombre):
    msg_salida = None
    msg_salida = random.choice(["Me agrada que quieras descubrir conocimiento, di una palabra clave (en inglés)\n", "Genial! dime una palabra clave (en inglés), para encontrar uno interesante\n", "Muy bien, busquemos uno interesante, dime una palabra clave que podría interesarte: \n"])
    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)

    subartic = []

    w = escuchar_mensaje(tunombre,w=True)
    print(chr(27)+"[1;34m"+'Buscando alguna coincidencia...')
    w = random.choice(w)
    try:
        subArtic = Artic[Artic['title'].str.contains(w)]
        # print(subArtic)
    #     if subartic != None:
        ran = np.random.randint(0,len(subArtic))
        title = Artic[['title']].iloc[ran][0]
        id = Artic[['id']].iloc[ran][0]
        msg_salida = (
            'Un artículo relacionado a {} que encontré para ti: {}'.format(w, GoogleTranslator(source='auto', target='es').translate(title)) +
             ' Puedes leerlo completo en: https://arxiv.org/abs/{}'.format(id))
    except:
        msg_salida = "Hum... creo que no capté algo de lo que dijiste, ¿podrías repetirlo?"
        print(chr(27)+"[1;34m"+'CHATBOT:')
        hablar(msg_salida)
        for i in textwrap.wrap(str(msg_salida), 130):
          print(chr(27)+"[1;34m"+ i)
        articulos(tunombre)
        msg_salida = []

    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)

    return

def wikis(tunombre):
    msg_salida = None
    msg_salida = random.choice(["¿Que te gustaría saber?, preguntame algún concepto\n", "Genial!, ¿te interesa saber la definición de algo en particular?\n", "Muy bien, podríamos empezar por algo que quieras saber... \n"])
    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)

    w = escuchar_mensaje(tunombre,w=True)
    w = GoogleTranslator(source='auto', target='en').translate(lista_a_cadena(w, ''))
    w = w.split()
    name = encontrar_en_lista(w, name_wikis)
    if name:
        ran = np.random.randint(0,len(Wikis[Wikis['Name']==name]))
        title = Wikis[Wikis['Name']==name][['Name', 'WikiDescription','WikiUrl']]
        msg_salida = (
            'Aquí esta la definición de {} que encontré para ti: {}'.format(
                GoogleTranslator(source='auto', target='es').translate(title.iloc[ran][0]), GoogleTranslator(source='auto', target='es').translate(lista_a_cadena(contar_puntos(title.iloc[ran][1]), ''))) +
             ' Puedes leer más de ello en: {}'.format(
                title.iloc[ran][2]))
    else:
        msg_salida = "Hum... creo que no capté algo de lo que dijiste, ¿podrías repetirlo?"
        print(chr(27)+"[1;34m"+'CHATBOT:')
        hablar(msg_salida)
        for i in textwrap.wrap(str(msg_salida), 130):
          print(chr(27)+"[1;34m"+ i)
        wikis(tunombre)
        msg_salida = []

    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)

    return

def investigadores(tunombre):
    msg_salida = None
    msg_salida = random.choice(["¿Que área del conocimiento te agrada en este momento?, estas son:\n", "Genial!, las áreas en las que podría encontrar a alguien son\n", "Muy bien, en que área estas interesado: \n"])
    categorias_inv = " - Fisica, Matemáticas y Ciencias de la Tierra\n - Biología y Química\n - Medicina y Ciencias de la Salud\n - Humanidades y Ciencias de la Conducta\n - Ciencias Sociales\n - Biotecnología y Ciencias Agropecuarias\n - Ingenierias"
    # msg_salida = msg_salida + categorias_inv
    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida+categorias_inv)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)
    print("\n"+chr(27)+"[1;34m"+'CHATBOT:\n'+categorias_inv)
    msg = escuchar_mensaje(tunombre)
    name = encontrar_en_lista(msg, LEER_INV)
    if name:
        ran = np.random.randint(0,len(Inv[Inv['Área del Conocimiento']==DIC_INV[name]]))
        datos = Inv[Inv['Área del Conocimiento']==DIC_INV[name]][['Nombre Completo', 'Área del Conocimiento', 'Institución de Adscripción']].iloc[ran]
        msg_salida = (
            'Si te gusta el área de {} yo te recomendaría contactar o buscar el trabajo desarrollado por "{}", adscrito a {}'.format(datos[1], datos[0], datos[2]))
    else:
        msg_salida = "Hum... creo que no capté algo de lo que dijiste, ¿podrías repetirlo?"
        print(chr(27)+"[1;34m"+'CHATBOT:')
        hablar(msg_salida)
        for i in textwrap.wrap(str(msg_salida), 130):
          print(chr(27)+"[1;34m"+ i)
        investigadores(tunombre)
        msg_salida = []

    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)


    return

def covids(tunombre):
    msg_salida = None
    msg_salida = random.choice(["Muy bien! Aquí hay algo de información sobre Covid. ¡Cuida tu salud!\n Te dejo los datos actualizados sobre covid en México\n Y un mapa interactivo de la evolución de covid en el mundo."])
    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)
    covid = Covid()
    casos = covid.get_status_by_country_name(country_name='mexico')
    print(chr(27)+"[1;31m"+'CHATBOT: Aquí hay un poco de información actualizada de Covid en México: \n')
    for x in casos:
        print(chr(27)+"[1;31m"+ x, ':', casos[x])
    creargrafica()

    return

### Switchs ###

def switcher_entretenimiento(key,tunombre):
    switch_entretenimiento = {
        "videos": videos,
        "peliculas": peliculas,
        "series": series,
        "musica": musica,
        "libros": libros,
        "videojuegos": juegos,
        "juegos": juegos
    }
    funcion = switch_entretenimiento.get(key)
    return funcion(tunombre)

def switcher_academico(key,tunombre):
    switch_academico = {
        "articulos": articulos,
        "articulo": articulos,
        "investigadores": investigadores,
        "investigador": investigadores,
        "definicion": wikis,
        "definiciones": wikis,
    }
    funcion = switch_academico.get(key)
    return funcion(tunombre)


### Casos 1###
def entretenimiento(tunombre):
    msg_salida = None
    msg_salida = random.choice(["¿Que te gustaría de entretenimiento?, tengo\n", "Genial!, tengo estas categorías\n", "Muy bien, entretenimiento, podríamos empezar por: \n"])
    categorias_entretenimiento = " - Videos\n - Películas\n - Series\n - Música\n - Libros\n - Videojuegos"
    # msg_salida = msg_salida + categorias_entretenimiento
    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida+categorias_entretenimiento)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)
    print("\n"+chr(27)+"[1;34m"+'CHATBOT:\n'+categorias_entretenimiento)
    msg = escuchar_mensaje(tunombre)
    key = encontrar_en_lista(msg, OP_ENTRETENIMIENTO)
    if key:
        switcher_entretenimiento(key,tunombre)
    else:
        msg_salida = "Hum... creo que no capté algo de lo que dijiste, ¿podrías repetirlo?"
        print(chr(27)+"[1;34m"+'CHATBOT:')
        hablar(msg_salida)
        for i in textwrap.wrap(str(msg_salida), 130):
          print(chr(27)+"[1;34m"+ i)
        entretenimiento(tunombre)
        msg_salida = []
    return

def academico(tunombre):
    msg_salida = None
    msg_salida = random.choice(["Tengo distintas recomendaciones académicas, algunas son:\n", "Genial!, tengo estas categorías\n", "Muy bien, el ámbito académico, podríamos empezar por: \n"])
    categorias_academico = " - Artículos\n - Investigadores\n - Definiciones"
    # msg_salida = msg_salida + categorias_academico
    print(chr(27)+"[1;34m"+'CHATBOT:')
    hablar(msg_salida+categorias_academico)
    for i in textwrap.wrap(str(msg_salida), 130):
      print(chr(27)+"[1;34m"+ i)
    print("\n"+chr(27)+"[1;34m"+'CHATBOT:\n'+categorias_academico)
    msg = escuchar_mensaje(tunombre)
    key = encontrar_en_lista(msg, OP_ACADEMICO)
    if key:
        switcher_academico(key,tunombre)
    else:
        msg_salida = "Hum... creo que no capté algo de lo que dijiste, ¿podrías repetirlo?"
        print(chr(27)+"[1;34m"+'CHATBOT:')
        hablar(msg_salida)
        for i in textwrap.wrap(str(msg_salida), 130):
          print(chr(27)+"[1;34m"+ i)
        academico(tunombre)
        msg_salida = []

    return

def switcher_general(key,tunombre):
    switch_general = {
        "entretenimiento": entretenimiento,
        "academico": academico,
        "covid": covids
    }

    funcion = switch_general.get(key)
    return funcion(tunombre)

def general(tunombre):
    msg = escuchar_mensaje(tunombre)
    primer_mensaje = encontrar_en_lista(msg, ["entretenimiento","academico","covid"])
    if primer_mensaje:
        switcher_general(primer_mensaje, tunombre)
    else:
        msg_salida = "Hum... creo que no capté algo de lo que dijiste, ¿podrías repetirlo?"
        print(chr(27)+"[1;34m"+'CHATBOT:')
        hablar(msg_salida)
        for i in textwrap.wrap(str(msg_salida), 130):
          print(chr(27)+"[1;34m"+ i)
        general(tunombre)
        msg_salida = []
    return

def chatear(path):
    """función principal para tener un chat."""
    getDatos(path)
    print(chr(27)+"[1;34m"+'Qué tál! Soy tu amigo MMN Bot! ¿Cuál es tu nombre?: \n')
    hablar('Qué tál! Soy tu amigo MMN Bot! ¿Cuál es tu nombre?:')
    chat = True
    tunombre = None
    with SRG.Microphone() as s:
            print('Estoy escuchando...')
            entrada_audio = st.record(s, duration=5)
            sys.stdout.write("\033[F")
            try:
                texto_salida = st.recognize_google(entrada_audio,language="es")
            except:
                print("No he podido escucharte, intenta de nuevo")
                texto_salida = escuchar_mensaje(tunombre)
    if tunombre != None:
      msg = texto_salida#input(chr(27)+"[1;30m"+str(tunombre) +': \t') \\ chr(27)+"[1;30m"+str(tunombre) +': \t' + texto_salida
      print(chr(27)+"[1;30m"+str(tunombre) +': \t' + msg)
    else:
      msg = texto_salida#input(chr(27)+"[1;30m"+'INPUT : \t')
      print(chr(27)+"[1;30m"+'INPUT : \t' + msg)
      n = msg.upper()   # n sirve para la función de nombres
      n = n.split()       # en lugar de la funcion preparar_texto
      for i in n:
        i = [i]
        if esta_en_lista(i, LEER_NOMBRES):
            tunombre = encontrar_en_lista(i, LEER_NOMBRES)
            msg_salida =(lista_a_cadena([tunombre.capitalize(),
                                        selector(i, LEER_NOMBRES, DECIR_NOMBRES)], ' '))
    while chat:
        msg = None
        msg_salida = None
        preg1 = random.choice(PREGUNTA_1)
        print(chr(27)+"[1;34m"+'CHATBOT: \t'+ preg1)
        hablar(preg1)

        general(tunombre)
        msg_salida = "Genial, un gusto hablar contigo ¿quieres continuar conversando?"
        print(chr(27)+"[1;34m"+'CHATBOT: \t'+ msg_salida)
        hablar(msg_salida)
        msg = escuchar_mensaje(tunombre)
        if terminar_chat(msg):
            msg_salida = 'Adios!'
            print(chr(27)+"[1;34m"+'CHATBOT: \t'+ msg_salida)
            hablar(msg_salida)
            chat = False
