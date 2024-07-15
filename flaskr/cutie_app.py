from os import path
import logging
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict
from hydra import compose, initialize
import gc
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from cutie.inference.data.vos_test_dataset import VOSTestDataset
from cutie.inference.data.burst_test_dataset import BURSTTestDataset
from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from cutie.inference.utils.results_utils import ResultSaver, make_zip
from cutie.inference.utils.burst_utils import BURSTResultHandler
from cutie.inference.utils.args_utils import get_dataset_cfg

import cv2

from tqdm import tqdm

import cv2
from cutie.gui.interactive_utils import image_to_torch, torch_prob_to_numpy_mask, index_numpy_to_one_hot_torch, overlay_davis


# default configuration
CONFIG = {
    'top_k': 30,
    'mem_every': 1,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
}

DEVICE = "cuda"

class Cutie:

    def __init__(self,num_obj = 0, propagation_frames = 100) -> None:
        with torch.inference_mode():
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            initialize(version_base='1.3.2', config_path="cutie/config", job_name="eval_config")
            self.cfg = compose(config_name="eval_config")

            with open_dict(self.cfg):
                self.cfg['weights'] = './weights/cutie-base-mega.pth'

            data_cfg = get_dataset_cfg(self.cfg)

            # Load the network weights
            self.cutie = CUTIE(self.cfg).cuda().eval()
            self.model_weights = torch.load(self.cfg.weights)
            self.cutie.load_weights(self.model_weights)

            self.current_frame_index = 1
            self.max_propagation_frames = propagation_frames
            



    def setMaxFrames(self, numFrames):
        self.max_frames = numFrames

    def setNumObj(self, num_obj):
        pass
        # self.procesor = InferenceCore(self.cutie, config = self.cfg)
        # self.procesor.set_all_labels(range(1, num_obj + 1 ))

    def _obtainIdsFromMask(self,mask):
        ids = []

    def propagate(self,frames,mask, num_obj,current_frame = 0):

        masks = []
        #print(mask)
        mask_aux = np.array(mask).astype(np.uint8)
        #mask_aux = cv2.resize(mask_aux,(1280,720))

        torch.cuda.empty_cache()

        processor = InferenceCore(self.cutie, cfg=self.cfg)

        #print(frames[0].shape)
        height,width, _ = frames[0].shape

        self.current_frame_index = current_frame

        frames_propagated = 0

        first_frame = True
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):

                while self.current_frame_index < self.max_frames and frames_propagated < self.max_propagation_frames:
                    frame = frames[self.current_frame_index]

                    print(f"frame: {self.current_frame_index}")
                    aux_frame = frame
    #<               aux_frame = cv2.resize(frame,(1280,720), interpolation=cv2.INTER_NEAREST_EXACT)
                    #print(image_to_torch(aux_frame, device=DEVICE))
                    frame_torch= image_to_torch(aux_frame, device=DEVICE)

                    #print(frame)
                    if first_frame:
                        mask_torch = index_numpy_to_one_hot_torch(mask_aux, num_obj +1).to(DEVICE)

                        prediction = processor.step(frame_torch, mask_torch[1:], idx_mask = False)
                        first_frame = False
                    else:
                        print(f"Mem info: {torch.cuda.mem_get_info()}")
                        prediction = processor.step(frame_torch)

                    prediction = torch_prob_to_numpy_mask(prediction)
                    
    #                prediction = cv2.resize(prediction,(width,height),interpolation=cv2.INTER_NEAREST_EXACT)
                    # cv2.imshow("ventana",prediction*255)
                    # cv2.waitKey(0)

    #                print(prediction)
                    #print(np.unique(prediction))
                    masks.append(prediction)    

                    self.current_frame_index += 1
                    frames_propagated += 1

                    torch.cuda.empty_cache()
                    gc.collect()
            #print(masks)
        
        return masks
    


    def backwards_propagate(self,frames,mask, num_obj,current_frame = 0):

        masks = []
        #print(mask)
        mask_aux = np.array(mask).astype(np.uint8)
        #mask_aux = cv2.resize(mask_aux,(1280,720))

        torch.cuda.empty_cache()

        processor = InferenceCore(self.cutie, cfg=self.cfg)

        #print(frames[0].shape)
        height,width, _ = frames[0].shape

        self.current_frame_index = current_frame

        frames_propagated = 0

        first_frame = True
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):

                while self.current_frame_index >= 0 and frames_propagated < self.max_propagation_frames:
                    frame = frames[self.current_frame_index]

                    print(f"frame: {self.current_frame_index}")
                    aux_frame = frame
    #<               aux_frame = cv2.resize(frame,(1280,720), interpolation=cv2.INTER_NEAREST_EXACT)
                    #print(image_to_torch(aux_frame, device=DEVICE))
                    frame_torch= image_to_torch(aux_frame, device=DEVICE)

                    #print(frame)
                    if first_frame:
                        mask_torch = index_numpy_to_one_hot_torch(mask_aux, num_obj +1).to(DEVICE)

                        prediction = processor.step(frame_torch, mask_torch[1:], idx_mask = False)
                        first_frame = False
                    else:
                        print(f"Mem info: {torch.cuda.mem_get_info()}")
                        prediction = processor.step(frame_torch)

                    prediction = torch_prob_to_numpy_mask(prediction)
                    
    #                prediction = cv2.resize(prediction,(width,height),interpolation=cv2.INTER_NEAREST_EXACT)
                    # cv2.imshow("ventana",prediction*255)
                    # cv2.waitKey(0)

    #                print(prediction)
                    #print(np.unique(prediction))
                    masks.append(prediction)    

                    self.current_frame_index -= 1
                    frames_propagated += 1

                    torch.cuda.empty_cache()
                    gc.collect()
            #print(masks)
        
        return masks