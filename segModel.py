import cv2
import numpy as np
# from cellpose import models
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from collections import defaultdict
import time
import sys

class segModel:
    def __init__(self, filename, save_filename=None) -> None:
        # self.model = models.Cellpose(gpu=True, model_type='cyto')
        self.filename = filename

        self.gray_stack, self.calcium_stack = self.get_frames()
        
        if save_filename is not None:
            self.save_filename = save_filename
        else:
            self.save_filename = self.filename.split('.mp4')[0]

    def get_frames(self):
        # Extract frames from video
        vidcap = cv2.VideoCapture(self.filename)
        success,image = vidcap.read()
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        gray_stack = np.zeros(shape=(image.shape[0],image.shape[1],frame_count))
        calcium_stack = np.zeros(shape=(image.shape[0],image.shape[1],frame_count))

        gray_stack[:,:,0] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        calcium_stack[:,:,0] = image[:,:,1]

        frame_idx = 1
        while success:
    
            success,image = vidcap.read()

            gray_stack[:,:,frame_idx] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            calcium_stack[:,:,frame_idx] = image[:,:,1]
            frame_idx+=1

        return gray_stack, calcium_stack

    def process_data(self, cell_diameter, flow_thresh, cell_prob_thresh, resample):

        start_time = time.time()

        # Load and average the frames in the video
        mean_mat = np.mean(self.gray_stack, axis=2)

        # Image Preprocessing

        # Enhance contrast
        processed_mean_mat = cv2.normalize(mean_mat, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # # Morphological closing
        kernel = np.ones((3,3),np.uint8)
        processed_mean_mat = cv2.morphologyEx(processed_mean_mat, cv2.MORPH_CLOSE, kernel,iterations = 1)

        # TODO: add a background subtraction?

        # Create the sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # Sharpen the image
        processed_mean_mat = cv2.filter2D(processed_mean_mat, -1, kernel)

        # Segment the image using cellpose
        masks, flows, styles, dia = self.model.eval(processed_mean_mat, diameter=cell_diameter, channels=[0, 0], cellprob_threshold=cell_prob_thresh, flow_threshold=flow_thresh, resample=resample)
        # diameter = None --> default diameter for 'cyto' model about somewhere around 20-30 pixels

        # Save out labeled image
        os.makedirs('results/', exist_ok=True)
        np.save('results/{}_segmentation.npy'.format(self.save_filename), masks)

        print(f'segmentation took {time.time()-start_time:.2f} seconds')

        return masks, mean_mat, processed_mean_mat

    # def get_mean_image_from_video(self):

    #     # Read video
    #     cap = cv2.VideoCapture(self.filename)
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    #     # Get frame width and height
    #     frame_width = int(cap.get(3))
    #     frame_height = int(cap.get(4))

    #     # Extract frames
    #     frame_mat = np.zeros(shape=(frame_height, frame_width, frame_count))
    #     frame_idx = 0
    #     while(cap.isOpened()):
    #         ret, frame = cap.read()
    #         if ret:
    #             # Convert frame to grayscale if it's not already
    #             if len(frame.shape) == 3:
    #                 gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #             else:
    #                 gray_frame = frame
    #             frame_mat[:,:,frame_idx] = gray_frame
    #         else:
    #             break
        
    #     # Take the average of the frames
    #     mean_mat = np.mean(frame_mat, axis=2)
        
    #     return mean_mat

    # def get_mean_image_from_tiffs(self):
    #     # images
    #     image_filenames = sorted(os.listdir(self.filename))

    #     im0 = cv2.imread(self.filename+'/'+image_filenames[0])

    #     frame_height = im0.shape[0]
    #     frame_width = im0.shape[1]
    #     frame_count = len(image_filenames)

    #     frame_mat = np.zeros(shape=(frame_height, frame_width, frame_count))
        
    #     for i in range(frame_count):
            
    #         frame = cv2.imread(self.filename+'/'+image_filenames[i])
            
    #         # Convert frame to grayscale if it's not already
    #         if len(frame.shape) == 3:
    #             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         else:
    #             gray_frame = frame
    #         frame_mat[:,:,i] = gray_frame

    #     # Take the average of the frames
    #     mean_mat = np.mean(frame_mat, axis=2)

    #     return mean_mat
    
    def visualize_segmentation(self, segmentation):
        # cap = cv2.VideoCapture(self.filename)

        # Define the codec and create VideoWriter object in .mp4 format
        # out_contours = cv2.VideoWriter('bot_01_before_test_contours_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height))

        # while(cap.isOpened()):
        #     # Capture frame-by-frame
        #     ret, frame = cap.read()

        #     break

        vidcap = cv2.VideoCapture(self.filename)
        success,frame = vidcap.read()

        for i in range(1, len(np.unique(segmentation)) + 1):
            mask = segmentation == i
            # if np.sum(mask) > 0:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # x, y, w, h = cv2.boundingRect(contours[0])
            cv2.drawContours(frame, contours, 0, (255, 0, 0), 1)
            # cv2.rectangle(boundary_image, (x, y), (x + w, y + h), (0, 0, 255), 1)

        plt.imshow(frame)
        plt.axis('off')
        plt.show()

    def save_segmentation_video(self, segmentation):
        # Read video
        vidcap = cv2.VideoCapture(self.filename)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # get fps from video
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print(f'{frame_count} frames in video')

        # Get frame width and height
        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))

        # Define the codec and create VideoWriter object in .mp4 format
        out_contours = cv2.VideoWriter('results/'+self.save_filename+'_segmentation.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height))
        frame_idx = 0

        contour_images = []

        success,frame = vidcap.read()

        while(success):
            # Capture frame-by-frame

            # retrieve the labeled image from data
            # draw contours
            
            # Draw contours
            contour_image = np.copy(frame)
            # boundary_image = np.copy(mean_mat)
            for i in range(1, len(np.unique(segmentation)) + 1):
                mask = segmentation == i
                if np.sum(mask) > 0:
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    # x, y, w, h = cv2.boundingRect(contours[0])
                    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)
                    # cv2.rectangle(boundary_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            
            out_contours.write(contour_image)
            # plt.imshow(contour_image)
            # plt.axis('off')
            # plt.savefig(f'results/{self.save_filename}_frame{frame_idx}.png', dpi=300, bbox_inches='tight')
            # plt.close()

            contour_images.append(contour_image)

            frame_idx += 1
            
            success, frame = vidcap.read()

        vidcap.release()
        out_contours.release()
        return contour_images, fps
    
    def extract_timeseries(self):
        pass

if __name__=='__main__':
    
    # Data
    FILENAME = '2022-10-13-6sLA-e 2-10s int.mp4' # bot_[id]_[phase]
    INPUT_TYPE = 0 # 0 = video (.mp4 file), 1 = folder of .tiff images

    # MODEL PARAMETERS

    DIAMETER = None # cell diameter in pixels
    # trained on 30 for cyto and 17 for nuclei
    # https://cellpose.readthedocs.io/en/latest/settings.html#diameter

    FLOW_THRESHOLD = 0.4
    # https://cellpose.readthedocs.io/en/latest/settings.html#flow-threshold

    CELL_PROB_THRESHOLD = 0.0 # also called 'mask threshold' in cellpose documentation
    # https://cellpose.readthedocs.io/en/latest/settings.html#mask-threshold

    RESAMPLE = False
    # https://cellpose.readthedocs.io/en/latest/settings.html#resample

    seg = segModel(filename=FILENAME)
    masks, mean_mat = seg.process_data(cell_diameter=DIAMETER, flow_thresh=FLOW_THRESHOLD, cell_prob_thresh=CELL_PROB_THRESHOLD, resample=RESAMPLE)
