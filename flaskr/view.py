import PySimpleGUI as sg
from io import BytesIO
from PIL import Image
from controller import Controller
import cv2
import numpy as np
import random
from PIL import ImageColor
import gc
from videoplayer import Videoplayer
from enum import Enum
from memory_profiler import profile 

GRAPH_WIDTH = 1280
GRAPH_HEIGHT = 768
PLAY_SPEED = 0.133

#Checking if git is correctly uploading the view file
CONST_VOID = -1

class Modo(Enum):
    NoneClickable = 0
    AddPoints = 1
    SelectAndModify = 2


class PathView:
    def __init__(self, theme="LightGreen") -> None:
        self.themeStr = theme
        # sg.theme(self.themeStr)
        # self._layout = [sg.FileBrowse(key = "-IN-"),sg.Button("Submit")]
        # self._window = sg.Window("Choose file",self._layout, modal = True,keep_on_top=True,finalize=True)


class Vista:
    def __init__(self, theme="LightGreen"):
        self._themeStr = theme

    #     sg.theme(self._themeStr)

        self._controller = Controller()

    #     self._layout = [self._getMenuLayout(), self._getContentLayout(), self._getButtonLayout()]

    #     self._window = sg.Window("Annotation app",self._layout,size=(1380,868), keep_on_top=False,finalize=False)
        self.pointMode = 1
        self.videoLoaded = False
        self.playing = False
        self.show_mask = True

    #     self.videoplayer = None
        self.color_table = self._generate_colors(2000)

        self.modo = Modo.NoneClickable

    #     self.zoom = 1
    #     self.zoomFocusPoint = (0,0)

        self.drag = False
        self.temporal_mask = []

        self.figures_ids = []
    # def _getButtonLayout(self):
    #     layout = [
    #         [sg.Button("<-- Past frame"), sg.Button("Apply SAM"),sg.Button("Apply Yolo"),sg.Button("Apply SAM full"), sg.Button("Next frame -->"),sg.Button("Change Mode"),sg.Button("Foreground"),sg.Button("Background")],
    #         [sg.Button("Propagate"),sg.Button("Backwards Propagate"),sg.Button("Backward"),sg.Button("Stop"),sg.Button("Play Forward"),sg.Button("Show Mask"),sg.Button("Clear selection"),sg.Button("Clear Frame"),sg.Button("Clear"),sg.Button("Zoom In"),sg.Button("Zoom Out"), sg.Slider(range=(0,100), default_value= 1, enable_events = True, orientation="horizontal", key="-SL-"),sg.Exit()]
    #     ]

    #     return(layout)
    
    # def _getMenuLayout(self):
    #     menu_def = [['&File', ['&Open video','&Open directory','&Save annotations','Load annotations']],['&Corrections', ['&Add Submask','&Add segmented line','&Remove submask', 'Remove segmented line']]]

    #     layout = [
    #         [sg.ButtonMenu("Menu", key="-BMENU-", menu_def=menu_def[0]),sg.ButtonMenu("Corrections", key="-BMENU-", menu_def=menu_def[1])],
            
    #     ]

    #     return layout

    # def _getContentLayout(self):
    #     layout = [
    #         [sg.Graph(canvas_size=(1280,768), graph_bottom_left=(0,0), graph_top_right=(1280,768), background_color = "white", enable_events=True,drag_submits=True,key="-GRAPH-")],
    #         #[sg.Image(key='-IMAGE-',enable_click_events=True)],
    #     ]

    #     return layout
    
    def _adapt_points(self,points,shape):

        factor_x = shape[1] / GRAPH_WIDTH
        factor_y = shape[0] / GRAPH_HEIGHT

        return (points[0] * factor_x, points[1] * factor_y)

    def res_conv(self, x: int, y: int, resolution_from: tuple, resolution_to: tuple) -> tuple:
        #assert x < resolution_from[0] and y < resolution_from[1], "Input coordinate is larger than resolution"
        y_ratio = resolution_to[0] / resolution_from[0] # ratio to multiply x co-ord
        x_ratio = resolution_to[1] / resolution_from[1] # ratio to multiply y co-ord
        return int(x * x_ratio), int(y * y_ratio)

    def _paint_points(self, frame):
        if self._controller.pointsExist():

            aux_frame = frame.copy()
            for point,label in zip(self._controller.getPoints(),self._controller.getLabels()):

                aux_point = (int(point[0]),int(point[1]))
                #aux_point = self.res_conv(aux_point[0],aux_point[1],self._controller.getCurrentShape(),(GRAPH_HEIGHT,GRAPH_WIDTH))
                if label == 1:
                    aux_frame = cv2.circle(aux_frame, aux_point, 5, (0,255,0), -1)
                else:
                    aux_frame = cv2.circle(aux_frame, aux_point, 5, (0,0,255), -1)
            return aux_frame
        return frame

    def _generate_colors(self, n):
        random.seed(10)
        return ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]

    def _apply_mask(self, image, mask, color, alpha=0.8):
        aux_image = image.copy()
        mask_aux = np.zeros(aux_image.shape[0:2])
        mask_aux = cv2.fillPoly(mask_aux, np.array([mask]).astype(np.int32), color=1)
        color = ImageColor.getrgb(color)

        for c in range(3):
            aux_image[:, :, c] = np.where(mask_aux == 1,
                                    aux_image[:, :, c] *
                                    (1 - alpha) + alpha * color[c],
                                    aux_image[:, :, c])
        return aux_image

    def _addSelectedMask(self, image, mask):
        if mask:
            aux_frame = image.copy()
            points = mask.getImportantPoints()
        return image

    def _addCurve(self, frame):

        if self.temporal_mask:
            if len(self.temporal_mask) > 1:
       
                aux_frame = frame.copy()
                points = np.asarray(self.temporal_mask,dtype=np.int32)
                points = points.reshape(-1,1,2)
                aux_frame = cv2.polylines(aux_frame, [points], False, (0,0,0), 5)
                return aux_frame
        return frame


    def _showFrame(self,frame, show_masks = True):
        results = frame.copy()

        if self.zoomFocusPoint[0] != -1:
            #results = self.zoom_at(results,self.zoomFocusPoint)
            pass

        if self.show_mask:
            masks = self._controller.getMasks()

            if masks:
                print(f"masks: {masks}")
                
                for mask in masks:
                    #print(mask.getId())
                    if not mask.selectedMask:

                        results = self._apply_mask(results,mask.getMask(),self.color_table[mask.getId()])
                    else:
                        if self.modo == Modo.SelectAndModify and not self.drag:
                            results = self._addSelectedMask(results,mask)
                        results = self._apply_mask(results,mask.getMask(),self.color_table[mask.getId()])
                
                if self.drag:
                    results = self._addCurve(results)
                    
        if self.modo == Modo.AddPoints:
            results = self._paint_points(results)


        #print(results)
        results = cv2.resize(results,(GRAPH_WIDTH,GRAPH_HEIGHT))
        data=cv2.imencode('.ppm',results)[1].tobytes()

        height, width, channels = results.shape

        graph = self._window["-GRAPH-"]
        #graph.erase()
        id = graph.draw_image(data=data, location=(0,height))
        self.figures_ids.append(id)
        print(self.figures_ids)
        if len(self.figures_ids) > 1:
            graph.delete_figure(self.figures_ids[0])
            self.figures_ids.pop(0)
        gc.collect()
        
        #image = self._window["-IMAGE-"]
        #image.update(data=cv2.imencode('.ppm',frame)[1].tobytes())
        
    # def changeZoomLvl(self,positive=True):
    #     if positive:
    #         self.zoom += 0.1
    #     else:
    #         self.zoom -= 0.1

    def zoom_at(self, img, coord=None):
        return img
        if self.zoom != 1:
            # Translate to zoomed coordinates
            h, w, _ = [ self.zoom * i for i in img.shape ]
            
            if coord is None: cx, cy = w/2, h/2
            else: cx, cy = [ self.zoom*c for c in coord ]
            
            img = cv2.resize( img, (0, 0), fx=self.zoom, fy=self.zoom)
            img = img[ int(round(cy - h/self.zoom * .5)) : int(round(cy + h/self.zoom * .5)),
                    int(round(cx - w/self.zoom * .5)) : int(round(cx + w/self.zoom * .5)),
                    : ]
            
            return img
        else:
            return img
    
    def _showFrameN(self,frame,mask = None):
        pass

    def _nextFrame(self, forward = True):
        if forward:
            newFrame = self._controller._showNextFrame()
        else:
            newFrame = self._controller._showLastFrame()
        self._showFrame(newFrame)

    def _loadVideo(self,path):
        firstFrame = self._controller._loadVideo(path)
        self._showFrameN(firstFrame)
        self.videoplayer = Videoplayer(PLAY_SPEED,self._nextFrame)
        self.videoplayer.stop()

    def _nextMode(self):
        self.modo = Modo((self.modo.value % 2) + 1)
   
    def _showFrameRefresh(self):
        frame = self._controller._showFrame()
        if frame is not None:
            self._showFrame(frame)

    # def run(self):
    #     while True:
    #         event, values = self._window.Read()
    #         print(values)
    #         print(event)
    #         if event in ('Exit',sg.WIN_CLOSED):
    #             self.videoplayer.stop()
    #             break
    #         elif event == "-BMENU-":
    #             if values['-BMENU-'] == 'Open video':
    #                 file = sg.popup_get_file("Choose File")
    #                 if file:
    #                     self._controller._loadVideo(file)
    #                     self._showFrame(self._controller._showFrame())
    #                     self.modo = Modo.AddPoints
    #                     self.videoplayer = Videoplayer(PLAY_SPEED,self._nextFrame)

    #             elif values['-BMENU-'] == 'Open directory':
    #                 file = sg.popup_get_folder("Choose Folder")
    #                 if file:
    #                     self._controller._loadVideo(file,directory=True)
    #                     self._showFrame(self._controller._showFrame())
    #                     self.modo = Modo.AddPoints
    #                     self.videoplayer = Videoplayer(PLAY_SPEED,self._nextFrame)

    #             elif values['-BMENU-'] == 'Save annotations':
    #                 print("Saving started")
    #                 self._controller.save_ann()
    #                 print("Finished saving")

    #             elif values['-BMENU-'] == 'Load annotations':
    #                 file = sg.popup_get_file("Choose File")
    #                 if file:
    #                     print("Loading started")

    #                     self._controller.load_ann(file)
    #                     print("Finished loading")

    #         elif event == "-BMENU-0":
    #             if values['-BMENU-0'] == "Add Submask":
    #                 if self._controller.getSelectedMasks() == 1:
    #                     print("Modifiying mask")
    #                     if self._controller.addMaskMod(self.temporal_mask):
    #                         self._showFrameRefresh()
    #                         self.temporal_mask = []
    #             elif values['-BMENU-0'] == "Add segmented line":
    #                 if self._controller.getSelectedMasks() == 1:
    #                     print("Modifiying mask")
    #                     if self._controller.addSegmentedLine(self.temporal_mask):
    #                         self._showFrameRefresh()
    #                         self.temporal_mask = []
    #             elif values['-BMENU-0'] == "Remove submask":
    #                 if self._controller.getSelectedMasks() == 1:
    #                     print("Modifiying mask")
    #                     if self._controller.delMaskMod(self.temporal_mask):
    #                         self._showFrameRefresh()
    #                         self.temporal_mask = []

    #             elif values['-BMENU-0'] == "Remove segmented line":
    #                 if self._controller.getSelectedMasks() == 1:
    #                     print("Modifiying mask")
    #                     if self._controller.delSegmentedLine(self.temporal_mask):
    #                         self._showFrameRefresh()
    #                         self.temporal_mask = []

    #         elif event == "Foreground":
    #             self.pointMode = 1
    #         elif event == "Background":
    #             self.pointMode = 0
    #             #print(self.pointMode)
    #         elif event == "Clear":
    #             self._controller.clearAll()
    #             self._showFrame(self._controller._showFrame()) 
    #         elif event == "Clear Frame":
    #             self._controller.clearSingleFrame()
    #             self._showFrame(self._controller._showFrame()) 
    #         elif event == "<-- Past frame":
    #             newFrame = self._controller._showLastFrame()
    #             self._showFrame(newFrame)
    #         elif event == "Next frame -->":
    #             newFrame = self._controller._showNextFrame()
    #             self._showFrame(newFrame)
    #         elif event == "Apply SAM":
    #             self._controller.applySAM()
    #             self._showFrame(self._controller._showFrame(),True)
    #         elif event == "Apply SAM full":
    #             self._controller.applySAM(allImage=True)
    #             self._showFrame(self._controller._showFrame(),True)
    #         elif event == "Propagate":
    #             self._controller.propagate()
    #             self._controller._showFrame()
    #         elif event == "Backwards Propagate":
    #             self._controller.backwards_propagate()
    #             self._controller._showFrame()
    #         elif event == "Show Mask":
    #             self.show_mask = not self.show_mask
    #             self._showFrame(self._controller._showFrame())
    #         elif event == "Play Forward":
                
    #             if self.videoplayer:
    #                 self.videoplayer.start()
    #         elif event == "Backward":
                
    #             if self.videoplayer:
    #                 self.videoplayer.change_forward()
    #                 self.videoplayer.start()
    #         elif event == "Stop":
    #             if self.videoplayer:
    #                 self.videoplayer.stop()
    #         elif event == "Change Mode":
    #             self._nextMode()
    #         elif event == "Apply Yolo":
    #             self._controller.applyYoloAndSam()
    #             self._showFrameRefresh()
    #         elif event == "Clear selection":
    #             self._controller.clearSelectedMasks()
    #             self._showFrameRefresh()
    #         elif event == "-SL-":
    #             if self.videoLoaded:
    #                 pass
    #         elif event == "-GRAPH-":
    #             point = values['-GRAPH-']
    #             point = self.res_conv(point[0], point[1], (GRAPH_HEIGHT,GRAPH_WIDTH), self._controller.getCurrentShape())
    #             point = (int(point[0]),self._controller.getCurrentShape()[0] - int(point[1]))
    #             if not self.modo == Modo.NoneClickable:
    #                 if values["-GRAPH-"][0] != None:
                        
    #                     if self.modo == Modo.AddPoints:

    #                         self._controller._addPoint(point,label = self.pointMode)
    #                         self._showFrame(self._controller._showFrame()) 
    #                     elif self.modo == Modo.SelectAndModify:

                            
    #                         if self.drag:
     
    #                             self.temporal_mask.append([point[0],point[1]])
       
    #                             self._showFrameRefresh()
    #                         elif not self.drag:
    #                             self._controller.selectMask(point)
    #                             self.drag = True
                
    #         elif event.endswith('+UP'):
                

    #             # if self._controller.getSelectedMasks() == 1 and self.drag:
    #             #     print("Modifiying mask")
    #             #     self._controller.addMaskMod(self.temporal_mask)

                
    #             self.drag = False
                
    #             print("Dragging ended")

    #     self._window.close()


# if __name__ == "__main__":

#     controller = Controller()
#     vista = Vista()
#     vista.run()
