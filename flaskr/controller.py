from model import Model
#from view import Vista

class Controller:
    def __init__(self, model = None, view = None) -> None:
        
        if model:
            self.model = model
        else:
            self.model = Model()
#        if view:
#            self.view = view
#        else:
#            self.view = Vista()

    def _showNextFrame(self):
        self.model._changeFrame(True)
        return self._showFrame()

    def _showLastFrame(self):
        self.model._changeFrame(False)
        return self._showFrame()

    def _showFrame(self):
        return self.model._getFrame()
       
    def _loadVideo(self,path,directory = False):
        self.model._loadVideo(path, directory)

    def _addPoint(self, point, label):
        self.model.addPoint(point,label)

    def applySAM(self, allImage = False):
        return self.model._applySAM(allImage)
    
    def getMasks(self):
        return self.model._getCurrentMasks()

    def getCurrentShape(self):
        return self.model.getCurrentShape()
    
    def pointsExist(self):
        return self.model.pointsExist()
    
    def getPoints(self):
        #print(f"Points: {self.model.getPoints()}")
        return self.model.getPoints()

    def getLabels(self):
        #print(f"Labels: {self.model.getLabels()}")
        return self.model.getLabels()
    
    def clearAll(self):
        self.model.clearAll()

    def clearSingleFrame(self):
        self.model.clearSingleFrame()
        
    def propagate(self):
        self.model.propagate()

    def backwards_propagate(self):
        self.model.backwards_propagate()

    def selectMask(self, point):
        self.model.selectMask(point)

    def save_ann(self):
        self.model.obtain_subimages()

    def load_ann(self, filePath):
        self.model.loadAnn(filePath)

    def getSelectedMasks(self):
        if self.model.videoLoaded():
            return self.model.getSelectedMasks()
        else:
            return 0
    
    def addMaskMod(self, maskToAdd):
        if self.model.videoLoaded():
            self.model.addMaskMod(maskToAdd)
            return True
        return False

    def addSegmentedLine(self, segmentToAdd):
        if self.model.videoLoaded():
            self.model.addSegmentedLine(segmentToAdd)
            return True
        return False

    def delSegmentedLine(self, segmentToDel):
        if self.model.videoLoaded():
            self.model.delSegmentedLine(segmentToDel)
            return True
        return False
    
    def delMaskMod(self, maskToDel):
        if self.model.videoLoaded():
            self.model.delMaskMod(maskToDel)
            return True
        return False
    
    def clearSelectedMasks(self):
        if self.model.videoLoaded():
            self.model.clearSelectedMasks()
            return True
        return False
    
    def applyYoloAndSam(self):
        if self.model.videoLoaded():
            self.model._applyYOLO()
            return True
        return False