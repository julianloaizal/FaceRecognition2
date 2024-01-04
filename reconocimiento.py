#OpenCV module
import cv2
#Modulo para leer directorios y rutas de archivos
import os
#OpenCV trabaja con arreglos de numpy
import numpy
#Se importa la lista de personas con acceso al laboratorio
from listaPermitidos import flabianos
flabs=flabianos()

# Parte 1: Creando el entrenamiento del modelo
print('Formando...')

#Directorio donde se encuentran las carpetas con las caras de entrenamiento
dir_faces = 'att_faces/orl_faces'

#Tamaño para reducir a miniaturas las fotografias
size = 4

# Crear una lista de imágenes y una lista de nombres correspondientes
(images, labels, names, id) = ([], [], {}, 0)
(im_width, im_height) = (112, 92)  # Tamaño estándar para las imágenes

# Directorio donde se encuentran las carpetas con las caras de entrenamiento
dir_faces = 'att_faces/orl_faces'

for subdir, dirs, files in os.walk(dir_faces):
    for subdir in dirs:
        subjectpath = os.path.join(dir_faces, subdir)
        names[id] = subdir  # Asignar el nombre del subdirectorio al id actual

        for filename in os.listdir(subjectpath):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Comprobar si el archivo es una imagen
                path = os.path.join(subjectpath, filename)
                img = cv2.imread(path, 0)  # Leer la imagen en escala de grises
                if img is not None:
                    img_resized = cv2.resize(img, (im_width, im_height))  # Redimensionar la imagen
                    images.append(img_resized)
                    labels.append(id)  # Usar el id actual como etiqueta

        id += 1  # Incrementar el id para el siguiente subdirectorio

# Imprimir el diccionario 'names' para depuración
print("Diccionario de nombres:", names)


print(f"Total de imágenes: {len(images)}")
print(f"Total de etiquetas: {len(labels)}")
print("Diccionario de nombres:", names)

# Crear una matriz Numpy de las dos listas anteriores
(images, labels) = [numpy.array(lis) for lis in [images, labels]]
# OpenCV entrena un modelo a partir de las imagenes
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)


# Parte 2: Utilizar el modelo entrenado en funcionamiento con la camara
face_cascade = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    #leemos un frame y lo guardamos
    rval, frame = cap.read()
    frame=cv2.flip(frame,1,0)

    #convertimos la imagen a blanco y negro    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #redimensionar la imagen
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    """buscamos las coordenadas de los rostros (si los hay) y
   guardamos su posicion"""
    faces = face_cascade.detectMultiScale(mini)
    
    for i in range(len(faces)):
        face_i = faces[i]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        # Intentado reconocer la cara
        prediction = model.predict(face_resize)
        
         #Dibujamos un rectangulo en las coordenadas del rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Escribiendo el nombre de la cara reconocida
        # La variable cara tendra el nombre de la persona reconocida
        cara = '%s' % (names[prediction[0]])

        #Si la prediccion tiene una exactitud menor a 100 se toma como prediccion valida
        if prediction[1]<100 :
          #Ponemos el nombre de la persona que se reconoció
          cv2.putText(frame,'%s - %.0f' % (cara,prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

          #En caso de que la cara sea de algun conocido se realizara determinadas accione          
          #Busca si los nombres de las personas reconocidas estan dentro de los que tienen acceso          
          #flabs.TuSiTuNo(cara)

        #Si la prediccion es mayor a 100 no es un reconomiento con la exactitud suficiente
        elif prediction[1]>101 and prediction[1]<500:           
            #Si la cara es desconocida, poner desconocido
            cv2.putText(frame, 'Desconocido',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))  

        #Mostramos la imagen
        cv2.imshow('OpenCV Reconocimiento facial', frame)

    #Si se presiona la tecla ESC se cierra el programa
    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyAllWindows()
        break
