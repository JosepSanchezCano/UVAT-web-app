import numpy as np
from ultralytics import SAM
import cv2
from memory_profiler import profile 
# from xmem import Xmem
from cutie_app import Cutie
import torch
import itertools
import gc
import glob
from ctypes import *
import os   
import create_annotations
import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from datetime import datetime
import math

import msretinex

from ultralytics import YOLO

IOU_THRES = 0.9

# so_file = "MSRetinex/MSR_original_lib.so"
# RetinexFunctions =  CDLL(so_file)
"""
    Clase Video: Contiene las estructuras de datos y las funciones necesarias para gestionar el
    video que debemos anotar. Entendemos como video un conjunto de frames

    frames       : array o lista de frames
    nframes      : número máximo de frames del video actual
    currentFrame : número que marca el frame actual del video

"""
class Video:
    def __init__(self, videoPath = None, directory = False) -> None:
        
        self.frames = []
        self.framesRetinex = []
        self.nframes = 0
        self.currentFrame = 0

        if videoPath:
            self._loadVideo(videoPath, directory)


    """ 
    Desc: Función que permite cargar el video en las estructuras de datos

    Params:
        - frames : variable temporal que permite cargar la lista de frames
        - video  : variable que almacena el objeto de opencv que almacena el video

    return : None
    """
    def _loadVideo(self, path, directory = False):
        frames = []
        retinex = []
        if directory:
            extensions = ('*.jpg')
            frames_paths = []
            for extension in extensions:
                frames_paths.extend(glob.glob(path+"/"+extension))

            #print(frames_paths)
            frames_paths.sort()
            frames_paths.pop(0)

            for file in frames_paths:
                frame = cv2.imread(file)
                frames.append(frame)

                ret = msretinex.mainRetinex(frame)
                print(f"the retinex image is: {ret}")
                retinex.append(ret)
                # retinex.append(msretinex.mainRetinex(frame))
            
        else:

            video = cv2.VideoCapture(path)
            while video.isOpened:
                read, frame = video.read()
                if not read:
                    break
                frames.append(frame)
                
                # ret = msretinex.mainRetinex(frame)
                #print(f"the retinex image is: {ret}")
                # retinex.append(ret)
            
        self.frames = frames
        self._setNFrames()


    """
    Desc: Función que fija el número máximo de frames

    return : None
    """
    def _setNFrames(self):
        self.nframes = len(self.frames)

    """ 
    Desc: Función getter que devuelve el frame actual
    """
    def _getCurrentFrame(self):
        gc.collect()
        return self.frames[self.currentFrame]
    
    """
    Desc: Función que cambia el frame actual, revisando si al aumentar o disminuir el frame nos salimos de los límites del video
    
    return: None
    """
    def _setFrame(self, nextFrame=True):
        gc.collect()
        if nextFrame and ((self.currentFrame + 1) < self.nframes):
            self.currentFrame += 1
        elif (self.currentFrame - 1) >= 0:
            self.currentFrame -= 1

    """
    Desc: Función que fija el frame actual a un número predeterminado, revisando que no supere los límites del video

    return: None
    """
    def _setFrameN(self, num):
        if num >= 0 and num < self.nframes:
            self.currentFrame = num

    """ 
    Desc: Función getter del frame actual
    """
    def _getCurrentFrameNumber(self):
        return self.currentFrame
    
    """
    Desc: Función getter de la lista de frames
    """
    def getFrames(self):
        return self.frames
    
    """ 
    Desc: Función que dado un número dentro de los límites del video devuelve un frame N
    
    return: Frame N
    """
    def _getNFrame(self, num):
        if num >= 0 and num < self.nframes:
            return self.frames[num]

"""
    Clase Mask: Clase del objeto máscara que permite gestionar la estructura de datos de una máscara.

    id  : Identificador de la máscara
    mask: Estructura de datos de la máscara en forma de lista de puntos que definen el polígono de la máscara
"""
class Mask:

    mask_ids = itertools.count()
    count_selected_masks = 0

    def __init__(self, mask, id = 0) -> None:
        if id == 0:
            self.id = next(Mask.mask_ids) + 1
        else:
            self.id = id

        self.mask = mask
        self.selectedMask = False

    # Función getter que devuelve la id de la máscara
    def getId(self):
        return self.id
    
    # Función setter que dado un número le asigna la id a la máscara
    def setId(self,id):
        self.id = id
    
    # Función getter que devuelve la lista de poligonos que forma la máscara
    def getMask(self):
        return self.mask
    
    @staticmethod
    def getSelectedMasks():
        return Mask.count_selected_masks
    
    @staticmethod
    def resetSelectedMasks():
        Mask.count_selected_masks = 0
    
    def addPointsToMask(self, points):
        #print(f"Mask: {self.mask}")
        for point in points:
            print(f"Adding point: {point}")
            self.mask.append(point)
        #print(f"Mask: {self.mask}")
        

    def changeMaskSelection(self, state = True):
        self.selectedMask = state
        if state:
            Mask.count_selected_masks += 1
        else:
            Mask.count_selected_masks -= 1

    def getImportantPoints(self):
        points = []
        idxs = []

        numPoints = len(self.mask)

        if numPoints > 5:
            for position, point in enumerate(self.mask):
                points.append(point)
                idxs.append(position)

        elif numPoints > 0:
            for position, point in enumerate(self.mask):
                points.append(point)
                idxs.append(position)

        return points,idxs
        


"""
    Clase SAMpred: Clase del predictor de SAM de la librería Ultralytics. La clase contiene el modelo, el camino
    a los pesos del mismo y las funciones para gestionar las predicciones del modelo

    weights : String que contiene la ruta a los pesos
    model   : Modelo SAM
"""
class SAMpred:
    def __init__(self, path = None, videoPath = None) -> None:
        
        self.weights = path
        self.model = None
        self._loadModel()

    # Función setter que fija el camino a los pesos del modelo    
    def _setWeights(self, path):
        self.weights = path

    # Función que carga el modelo en memoria. Actualmente no usa los pesos del atributo weights
    def _loadModel(self, SegmAM = True):
        if SegmAM:
            self.model = SAM("sam_b.pt")
        else:
            pass

    # Función que descarga el modelo de la memoria, tanto ram como de la GPU
    def unloadModel(self):
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    # Función que aplica la predicción del modelo, actualmente 
    def _applyPred(self, frame, points = None,labels = None, SAM = True, allImage = False, bbox = None):
        if SAM:
            if allImage:
                results = self.model(frame,verbose=False)
            else:
                if bbox:
                    results = self.model(frame, bboxes=bbox,verbose=False)
                else:
                    results = self.model(frame,points=points,labels = labels,verbose=False)
            
            return results

    
class Yolopred():
    def __init__(self, load = False) -> None:
        self.model = None

        if load:
            self.loadModel()

    def loadModel(self, path = "weights/bbox.pt"):
        self.model = YOLO(path)
        #print(f"El modelo es: {self.model}")

    def unloadModel(self):
        self.model = None
        torch.cuda.empty_cache()

    def applyPred(self,img):
        if self.model:
            pred = self.model(img)
            #print(pred)
            return pred

""" 
Clase Model: Clase que contiene la lógica necesaria y el resto de objetos de la clase para poder aplicar el modelo
a las imágenes del video que anotamos. 

sam           : Objeto SAM para aplicar prediciones (falta cambiarlo a una nomenclatura más generica cuando implementmos más modelos)
video         : Objeto que contiene el video que estamos anotando
currentPoints : Diccionario que para cada frame contiene los puntos que ha anotado el usuario
labels        : Diccionario que para cada frame contiene si el punto es de background o foreground
xmem          : Objeto genérico que representa el objeot que propaga las máscaras, actualmente cutie
"""
class Model:
    def __init__(self, videoPath = None) -> None:
        self.sam = SAMpred()
        self.video = None
        self.masks = dict()
        self.currentPoints = dict()
        self.labels = dict()
        self.xmem = Cutie()
        self.yolo = Yolopred()

        # Si tenemos camino al video al inicializar el modelo cargamos el video en memoria
        if videoPath:
            self._loadVideo(videoPath)

        
    #Función que carga el video, generando un objeto de la clase video y que fija el número máximo de frames
    def _loadVideo(self, path, directory=False):
        self.video = Video(path, directory)
        self._setMaxFrames()

    #Función que fija el número de frames máximo en función del video
    def _setMaxFrames(self):
        self.xmem.setMaxFrames(self.video.nframes)

    #Función que cambia el frame actual
    def _changeFrame(self,forward):
        self.video._setFrame(forward)

    #Función que añade un punto para SAM, asignandolo con un label donde 1 representa foreground y 0 representa background
    def addPoint(self, point, label = 1):

        key = self.video._getCurrentFrameNumber()

        if key not in self.currentPoints:
            self.currentPoints[key] = []
            self.labels[key] = []


        self.currentPoints[key].append(point)
        self.labels[key].append(label)

    #Función getter que devuelve el frame actual dentro del objeto video
    def _getFrame(self):
        return self.video._getCurrentFrame()

    #Función que dado dos listas de puntos, genera una lista donde las coordenadas x e y quedan emparejadas en tuplas
    def generate_polygon_from_det(self,x,y):

        if len(x) == len(y):
            temp = []
            for i in range(len(x)):
                temp.append([x[i],y[i]])
            return temp

    #Función que corrige las máscaras adapatandolas a un formato de array con tuplas de coordenadas que forman el poligono de la máscara
    def _correctMasks(self,masks):
        temp_masks = []

        for mask in masks:
            print(type(mask))
            print(mask.shape)
            temp_masks.append(self.generate_polygon_from_det(mask[:,0],mask[:,1]))
        return temp_masks
    
    #FUnción que devuelve los puntos de SAM para el frame que se esta visualizando
    def getPoints(self):
        if self.currentPoints:
            if self.video._getCurrentFrameNumber() in self.currentPoints:
                return self.currentPoints[self.video._getCurrentFrameNumber()]
            
    #Función que para el frame que se esta visualizando devuelve las etiquetas de los puntos
    def getLabels(self):
        if self.labels:
            if self.video._getCurrentFrameNumber() in self.labels:
                return self.labels[self.video._getCurrentFrameNumber()]
            
    #Función que aplica a la imagen actual SAM usando puntos o bounding boxes como input
    def _applySAM(self, allImage = False):
        if not allImage:
            points = self.getPoints()
            labels = self.getLabels()
        else:
            points = None
            labels = None

        if not allImage and points:
        #Aplicamos el modelo
            masks = self.sam._applyPred(self.video._getCurrentFrame(), points = points, labels = labels, allImage=allImage)[0].masks.xy
            # print(masks)
            #Adaptamos el output al formato que más nos conviene
            print(type(masks))
            masks = self._correctMasks(masks)
            print(type(masks))
            #Obtenemos el número del frame actual para usar como llave en el diccionario
            key = self.video._getCurrentFrameNumber()
            
            if key not in self.masks:
                self.masks[key] = []
            
            for mask in masks:

                self.masks[key].append(Mask(mask))

            self.fixMasks()

    def _applyYOLO(self):
        #Aplicamos el modelo
        self.yolo.loadModel()
        masks = self.yolo.applyPred(self.video._getCurrentFrame())[0].boxes.data.cpu().numpy()
        # print(masks.shape)
        # masks = self._correctMasks(masks)
        # print(masks)

        coordinates = []
        for mask in masks:
            # print(f"mask: {mask}")
            if mask[5] == 0 and mask[4] > 0.5:
                coordinates.append(mask[0:4])

        print(coordinates)

        masks_from_bbox = self.sam._applyPred(self.video._getCurrentFrame(),bbox = coordinates)[0].masks.xy
        # print(masks_from_bbox)
        print(type(masks_from_bbox))
        masks_from_bbox = self._correctMasks(masks_from_bbox)
        print(type(masks_from_bbox))
        
        #Adaptamos el output al formaro que más nos conviene
        # masks_from_bbox = np.int32([masks_from_bbox])
        #Obtenemos el número del frame actual para usar como llave en el diccionario
        key = self.video._getCurrentFrameNumber()
        
        if key not in self.masks:
            self.masks[key] = []
        
        for mask in masks_from_bbox:
            print(f"mask shape: {mask}")
            auxMask = list(mask)
            self.masks[key].append(Mask(auxMask))

        self.fixMasks()
        
        self.yolo.unloadModel()
    #Función que llama al objeto XMEM o similares para propagar las máscaras del frame actual al resto del video
    def propagate(self):
        #print(f"masks:\n{self.masks[self.video._getCurrentFrameNumber()][0].getMask()}")
        num_obj, mask = self.applyMaskAndFormatXmem()
        if num_obj != -1:
            self.xmem.setNumObj(num_obj)
            temp_masks = self.xmem.propagate(self.video.getFrames(),mask, num_obj,self.video._getCurrentFrameNumber())
            self.adaptMasks(temp_masks)
            self.fixAllMasks()

    def backwards_propagate(self):
        #print(f"masks:\n{self.masks[self.video._getCurrentFrameNumber()][0].getMask()}")
        num_obj, mask = self.applyMaskAndFormatXmem()
        if num_obj != -1:
            self.xmem.setNumObj(num_obj)
            temp_masks = self.xmem.backwards_propagate(self.video.getFrames(),mask, num_obj,self.video._getCurrentFrameNumber())
            self.adaptMasks(temp_masks, is_backwards = True)
            self.fixAllMasks()

    #Función que corta las máscaras obtenidas del modelo XMEM
    def cutMasks(self, temp_masks, is_backwards = False):
        masks = dict()
        key = self.video._getCurrentFrameNumber()

        #Para cada máscara aplicamos el proceso de cortar
        for mask in temp_masks:
            ids = np.unique(mask)

            #Creamos una lista vacia para las máscaras cortadas
            masks[key] = []
            for i in ids:
                if i != 0:
                    #Obtenemos una máscara booleana en función de la id de cada máscara
                    new_mask = (mask == i).astype(np.uint8)
                    
                    #Generamos un nuevo ovjeto máscara usando la máscara booleana
                    new_mask_obj = Mask(new_mask,i)
                    masks[key].append(new_mask_obj)

            if is_backwards:
                key -= 1 
            else:
                key += 1

        return masks
        
    #Función que adapta las máscaras obtenidas por los métodos XMEM 
    def adaptMasks(self, temp_masks, is_backwards = False):

        all_masks = self.cutMasks(temp_masks, is_backwards)

        for key, masks in all_masks.items():

            if key not in self.masks:
                self.masks[key] = []
  
            #Generamos las ids y las máscaras usando la función que genera las máscaras usando las máscaras binarias
            current_masks,ids = self._fromMask2Poly(masks)
            
            for mask,id in zip(current_masks,ids):

                self.masks[key].append(Mask(mask,id))

    #Función que transforma máscaras binarias a un conjuntos de puntos
    def _fromMask2Poly(self, masks, id = -1):
        new_masks = []
        ids = []
        for mask in masks:

            if not mask is None:
                if isinstance(mask,Mask):
                    aux_mask = mask.getMask().copy()
            
                    countours, hier = cv2.findContours(aux_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    #print(countours[0])
                    #new_masks.append()
                else:
      
                    aux_mask = mask.copy()
                 
                    countours, hier = cv2.findContours(aux_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                aux_list = []
                #print(countours)
           
                if countours:
                    countour = max(countours, key = cv2.contourArea)
                    
          
                    for item in countour.tolist():
        
                        for element in item:
                
                            aux_list.append(element)

                    new_masks.append(aux_list)
                    if id == -1:
                        ids.append(mask.getId())
                
        #print(new_masks) 
        return new_masks,ids

    #Función getter que devuelve las máscaras del frame actual
    def _getCurrentMasks(self):
        if self.video._getCurrentFrameNumber() in self.masks:
            return self.masks[self.video._getCurrentFrameNumber()]
        
    def _generateMasks(self):
        pass

    #Función getter que devuelve la forma actual del frame
    def getCurrentShape(self, frame = -1):
        if frame == -1:
            if self.video:
                return self.video._getCurrentFrame().shape
        else:
            return self.video._getNFrame(frame).shape
    
    #Función booleana que comprueba si existen puntos en el frame actual
    def pointsExist(self):
        if self.video._getCurrentFrameNumber() in self.currentPoints:
            return True
        return False
    
    #Función getter que devuelve todos los puntos del frame actual
    def videoLoaded(self):
        if self.video:
            return True
        else:
            return False
    
    #Función getter que devuelve los labels o etiquetas que indican si el punto es background o foreground
  
    #Función que reinicia todos los puntos, etiquetas y máscaras
    def clearAll(self):
        self.currentPoints = dict()
        self.labels = dict()
        self.masks = dict()
        Mask.resetSelectedMasks()
        
    #Función que limpia los puntos, etiquetas y máscaras del frame actual
    def clearSingleFrame(self):
        currentFrame = self.video._getCurrentFrameNumber()
        
        self.currentPoints[currentFrame] = []
        self.labels[currentFrame] = []
        self.masks[currentFrame] = []

    #Función que aplica la máscara a una imagen para preparar el uso de XMEM o similares
    def applyMaskAndFormatXmem(self):
        if self.masks:
            masks = self.masks[self.video._getCurrentFrameNumber()]
            image = self.video._getCurrentFrame()       
    
            mask_aux = np.zeros(image.shape[0:2])
            for mask in masks:
                # print(f"color id: {mask.getId()} type: {type(mask.getId())}")
                mask_aux = cv2.fillPoly(mask_aux, np.array([mask.getMask()]).astype(np.int32), color=int(mask.getId()))


            mask_aux = mask_aux * 255
            # cv2.imshow("ventana",image)
            # cv2.waitKey(0) 
            mask_aux = cv2.threshold(mask_aux, 127, 255, cv2.THRESH_BINARY)[1]
            mask_aux = mask_aux.astype(np.uint8)

            n1, regions = cv2.connectedComponents(mask_aux)
            return (n1-1), regions
        return -1,[]
    
    #Función que devuelve el conjunto de máscaras para el frame actual que han sido seleccionadas
    def getSelectedMasks(self):
        counter = 0
        if self.video._getCurrentFrameNumber() in self.masks:
            masks = self.masks[self.video._getCurrentFrameNumber()]
            for mask in masks:
                if mask.selectedMask:
                    counter += 1
        return counter

    #Función que borrar las máscaras selecionadas
    def clearSelectedMasks(self):
        counter = 0
        if self.video._getCurrentFrameNumber() in self.masks:
            masks = self.masks[self.video._getCurrentFrameNumber()]
            for count, mask in enumerate(masks):
                if mask.selectedMask:
                    self.popMask(count)
        return counter
    
    #Función que devuelve el conjunto de máscaras para el frame que se esta visualizando
    def getCurrentMasks(self):
        if self.video._getCurrentFrameNumber() in self.masks:
            return self.masks[self.video._getCurrentFrameNumber()]
        
    def getNMasks(self, position):
        if position in self.masks:
            return self.masks[position]

    def setMask(self, mask, position, posFrame = -1):
        if posFrame == -1:
            if self.masks[self.video._getCurrentFrameNumber()]:
                #print(f"Checking if position: {position} in masks: {self.masks[self.video._getCurrentFrameNumber()]}" )
                if position < len(self.masks[self.video._getCurrentFrameNumber()]):
                    #print("Set mask")
                    self.masks[self.video._getCurrentFrameNumber()][position] = mask
        else:
            if self.masks[posFrame]:
                if position < len(self.masks[posFrame]):
                    self.masks[posFrame][position] = mask

    def popMask(self, position, posFrame = -1):
        if posFrame == -1:
            if self.masks[self.video._getCurrentFrameNumber()]:
                #print(f"Checking if position: {position} in masks: {self.masks[self.video._getCurrentFrameNumber()]}" )
                if position < len(self.masks[self.video._getCurrentFrameNumber()]):
                    #print("popping")
                    self.masks[self.video._getCurrentFrameNumber()].pop(position)
        else:
            if self.masks[posFrame]:
                # print(f"Checking if position: {position} in masks: {self.masks[self.video._getCurrentFrameNumber()]}" )
                
                if position < len(self.masks[posFrame]):
                    # print("popping")
                    self.masks[posFrame].pop(position)

    def fuseMasks(self, p1, p2, posFrame = -1):
        print("fusing masks")
        if posFrame == -1:
            if self.masks[self.video._getCurrentFrameNumber()]:
                l1 = len(self.masks[self.video._getCurrentFrameNumber()])

                if p1 < l1 and p2 < l1 :
                    aux_image = np.zeros(self.getCurrentShape()[0:2])
                    
                    m1 = self.masks[self.video._getCurrentFrameNumber()][p1]
                    m2 = self.masks[self.video._getCurrentFrameNumber()].pop(p2)
                    
                    
                    #print(m1.getMask())
                    aux_image = cv2.fillPoly(aux_image,[np.array(m2.getMask()).astype(np.int32)], color=1)
                    aux_image = cv2.fillPoly(aux_image,[np.array(m1.getMask()).astype(np.int32)], color=1)

                    aux_image = (aux_image != 0).astype(np.uint8)

                    aux_list = []
                    aux_list.append(aux_image)

                    list_of_points_mask = self._fromMask2Poly(aux_list,id=m1.getId())

                    aux_mask = Mask(list_of_points_mask[0][0],m1.getId())

                    self.setMask(aux_mask,p1)
                    #self.popMask(p2)

        else:
            if self.masks[posFrame]:
                l1 = len(self.masks[posFrame])
                if p1 < l1  and p2 < l1:
                    aux_image = np.zeros(self.getCurrentShape(posFrame)[0:2])
                    
                    m1 = self.masks[posFrame][p1]
                    m2 = self.masks[posFrame].pop(p2)

                    
                    
                    aux_image = cv2.fillPoly(aux_image,[np.array(m2.getMask()).astype(np.int32)], color=1)
                    aux_image = cv2.fillPoly(aux_image,[np.array(m1.getMask()).astype(np.int32)], color=1)

                    aux_image = (aux_image != 0).astype(np.uint8)

                    aux_list = []
                    aux_list.append(aux_image)

                    list_of_points_mask = self._fromMask2Poly(aux_list,id=m1.getId())

                    aux_mask = Mask(list_of_points_mask[0][0],m1.getId())

                    self.setMask(aux_mask,p1)
                    #self.popMask(p2)


    def addMaskMod(self, mod):
        if len(mod) > 3:
            print("Trying to add mod")
            masks = self.getCurrentMasks()
      
            if masks is not None:
                for count, mask in enumerate(masks):

                    if mask.selectedMask:
                        print(f"Added mod in mask: {count}")
     
                        aux_image = np.zeros(self.getCurrentShape()[0:2])

                        print(np.array(mask.getMask()).astype(np.int32))
                        aux_image = cv2.fillPoly(aux_image, [np.array(mask.getMask()).astype(np.int32)], color = 1)
                        aux_image = cv2.fillPoly(aux_image, [np.array(mod).astype(np.int32)], color = 1)

                        aux_image = (aux_image != 0).astype(np.uint8)
                        # cv2.imshow("Ven",aux_image*255)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        aux_list = []
                        aux_list.append(aux_image)
                        list_of_points_mask = self._fromMask2Poly(aux_list,id = mask.getId())

                        aux_mask = Mask(list_of_points_mask[0][0],mask.getId())
                        
                        self.setMask(aux_mask,count)

    def addSegmentedLine(self, mod):
        if len(mod) > 3:
            print("Trying to add line")
            masks = self.getCurrentMasks()
      
            if masks is not None:
                for count, mask in enumerate(masks):

                    if mask.selectedMask:
                        print(f"Added mod in mask: {count}")
     
                        aux_image = np.zeros(self.getCurrentShape()[0:2])

          
                        aux_image = cv2.fillPoly(aux_image, [np.array(mask.getMask()).astype(np.int32)], color = 1)
                        aux_image = cv2.polylines(aux_image, [np.array(mod).astype(np.int32)], False, 1, 10)

                        aux_image = (aux_image != 0).astype(np.uint8)

                        aux_list = []
                        aux_list.append(aux_image)
                        list_of_points_mask = self._fromMask2Poly(aux_list,id = mask.getId())

                        aux_mask = Mask(list_of_points_mask[0][0],mask.getId())
                        
                        self.setMask(aux_mask,count)

    def delMaskMod(self, mod):
        if len(mod) > 3:
            print("Trying to del line")
            masks = self.getCurrentMasks()
      
            if masks is not None:
                for count, mask in enumerate(masks):

                    if mask.selectedMask:
                        print(f"Added mod in mask: {count}")
     
                        aux_image = np.zeros(self.getCurrentShape()[0:2])

                        print(np.array(mask.getMask()).astype(np.int32))
                        aux_image = cv2.fillPoly(aux_image, [np.array(mask.getMask()).astype(np.int32)], color = 1)
                        aux_image = cv2.fillPoly(aux_image, [np.array(mod).astype(np.int32)], color = 0)

                        aux_image = (aux_image != 0).astype(np.uint8)
                        cv2.imshow("Ven",aux_image*255)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        aux_list = []
                        aux_list.append(aux_image)
                        list_of_points_mask = self._fromMask2Poly(aux_list,id = mask.getId())

                        aux_mask = Mask(list_of_points_mask[0][0],mask.getId())
                        
                        self.setMask(aux_mask,count)

    def delSegmentedLine(self, mod):
        if len(mod) > 3:
            print("Trying to del line")
            masks = self.getCurrentMasks()
      
            if masks is not None:
                for count, mask in enumerate(masks):

                    if mask.selectedMask:
                        print(f"Added mod in mask: {count}")
     
                        aux_image = np.zeros(self.getCurrentShape()[0:2])

                        print(np.array(mask.getMask()).astype(np.int32))
                        aux_image = cv2.fillPoly(aux_image, [np.array(mask.getMask()).astype(np.int32)], color = 1)
                        aux_image = cv2.polylines(aux_image, [np.array(mod).astype(np.int32)], False, 0, 10)

                        aux_image = (aux_image != 0).astype(np.uint8)
                        # cv2.imshow("Ven",aux_image*255)
                        # cv2.waitKey(0)
                        aux_list = []
                        aux_list.append(aux_image)
                        list_of_points_mask = self._fromMask2Poly(aux_list,id = mask.getId())

                        aux_mask = Mask(list_of_points_mask[0][0],mask.getId())
                        
                        self.setMask(aux_mask,count)

    def selectMask(self, point):
        if self.masks:
            
            masks = self.masks[self.video._getCurrentFrameNumber()]

            pointAux = Point(point)
            

            # foundMask = False
            # mask = None
            print(f"Point for selection of mask: {pointAux}")
            for counter,mask in enumerate(masks):
                polygon = Polygon(mask.getMask())
                #print(f"Polygon: {polygon}")
                if polygon.contains(pointAux):

                    masks[counter].changeMaskSelection(True)
                    print("Mask selected")
                    break
        return 0
    
    def bounding_box(self,points):
        x_coor, y_coor = zip(*points)

        return min(x_coor),min(y_coor),max(x_coor), max(y_coor)

    def obtain_subimages(self):
        annotation_id = 0
        coco_format = create_annotations.get_coco_json_format()
        category_ids = {
        "peix": 1,
    }
        time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        basename = "video_" + time
        image_annotations = []
        images = []
        coco_format["categories"] = create_annotations.create_category_annotation(category_ids)

        os.mkdir(basename)

        for key,value in self.masks.items():
            # print(f"type of key is : {type(key)} and obj is : {key}")
            # load image info
            # info = dataset.image_info[image_id]
            # image = dataset.load_image(image_id)
            image = self.video._getNFrame(key)
            # print(info)
            height = image.shape[0]
            width = image.shape[1]

            imageA = create_annotations.create_image_annotation(file_name = basename + "_" + str(annotation_id),width=width,height=height)
            images.append(imageA)
            cv2.imwrite("/"+basename+"/"+basename + "_" + str(annotation_id)+".png",image)

            for mask in value:
                # print(f"type of Id is : {type(mask.getId())} and obj is : {mask.getId()}")
                x1,y1,x2,y2 = self.bounding_box(mask.getMask())

                mask_poly = np.zeros(image.shape[0:3])
                mask_poly = cv2.fillPoly(mask_poly, np.array([mask.getMask()]).astype(np.int32), color=(0,255,0))

                contours = create_annotations.find_contours(mask_poly.astype(np.uint8))
                contour = max(contours, key = cv2.contourArea)
                # for contour in contours:
                annotation = create_annotations.create_annotation_format(contour,key,imageA['id'],category_ids['peix'],int(mask.getId()),(int(x1),int(y1),int(x2-x1),int(y2-y1)))
                annotation_id += 1
                image_annotations.append(annotation)

            # print(image_annotations)
        coco_format["images"] = images
        coco_format["annotations"] = image_annotations

        with open(f"output/ann_{time}.json", "w+") as outfile:
                json.dump(coco_format, outfile, sort_keys=True, indent=4)

        return None
    
    def _generateMaskFromAnn(self, list_of_points):
        auxList = []

        counter = 0
        print(list_of_points)
        while counter < len(list_of_points):
            auxList.append([list_of_points[counter],list_of_points[counter+1]])
            counter += 2        
            
        return auxList

    def loadAnn(self, annFile):
        self.currentPoints = dict()
        self.labels = dict()
        self.masks = dict()

        jsonFile = open(annFile)
        jsonObj = json.load(jsonFile)

        anns = jsonObj["annotations"]

        for item in anns:
            mask_as_points = self._generateMaskFromAnn(item["segmentation"][0])
            print(mask_as_points)
            aux_mask = Mask(mask_as_points,item["id"])
            if item["frame_number"] in self.masks:
                self.masks[item["frame_number"]].append(aux_mask)
            else:
                self.masks[item["frame_number"]] = []
                self.masks[item["frame_number"]].append(aux_mask)
                        
    
    def _isBigger(self,m1, m2):
        m1Area = np.count_nonzero(m1 == 1)
        m2Area = np.count_nonzero(m2 == 1)
        
        return m1Area > m2Area
            
    # Intersection max, fix function
    def _calcIou(self, mask1, mask2):

        binarymask1 = np.zeros(self.getCurrentShape())
        binarymask2 = np.zeros(self.getCurrentShape())
        

        binarymask1 = cv2.fillPoly(binarymask1, np.array([mask1.getMask()]).astype(np.int32), color=1)
        binarymask2 = cv2.fillPoly(binarymask2, np.array([mask2.getMask()]).astype(np.int32), color=1)

        big = self._isBigger(binarymask1,binarymask2)

        binarymask1 = torch.Tensor(binarymask1)
        binarymask2 = torch.Tensor(binarymask2)

        intersection = (binarymask1 * binarymask2).sum()
        if intersection == 0:
            return 0,False
        
        A = binarymask1.sum()
        B = binarymask2.sum()
        
        # union = torch.logical_or(binarymask1, binarymask2).to(torch.int).sum()

        aux_A = intersection / A
        aux_B = intersection / B

        max = np.max([aux_A,aux_B])

        return max, big

    def fixAllMasks(self):
        for i in range(len(self.masks)):
            self.fixMasks(pos = i)

    def reducePoints(self, listOfPoints, minDis = 3):

        newPoints = []
        newPoints.append(listOfPoints[0])

        for point in listOfPoints:
            if math.dist(newPoints[-1],point) >= minDis:
                newPoints.append(point)

    def fixMasks(self, pos = -1):
    
        if pos == -1:
            masks = self.getCurrentMasks()
        else:
            masks = self.getNMasks(pos)
        #print(masks)
            
        if masks:
            for count, mask1 in enumerate(masks):
                #print(f"count: {count} and maskref: {mask1}")
                for element, mask2 in enumerate(masks):
                    if element != count:
                        #print(f"element: {element} and maskref: {mask2}")
                        iou, itemToRemove = self._calcIou(mask1,mask2)
                        # print(f"IoU: {iou} between {count} and {element}")
                        if iou >= IOU_THRES:
                            if itemToRemove:
                                self.fuseMasks(count,element, pos)
                            else:
                                self.fuseMasks(element,count, pos)
                                continue
