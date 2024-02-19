import cv2
import numpy as np
from cellpose import models
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

class segModel:
    def __init__(self, filename, save_filename=None, gpu=True, mode=None) -> None:
        self.model = models.Cellpose(gpu=gpu, model_type='cyto')
        self.filename = filename
        
        if mode is None:
            print("Specify segmentation mode: stationary, non-stationary, do3D") # TODO: change this once non-stationary method has been nailed down
        self.mode = mode

        self.gray_stack, self.calcium_stack = self.get_frames()
        
        if save_filename is not None:
            self.save_filename = save_filename
        else:
            self.save_filename = self.filename.split('.mp4')[0]

    def get_frames(self):
        # Extract frames from video
        vidcap = cv2.VideoCapture(self.filename)
        success,image = vidcap.read()
        # frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # TESTING: use only the first few frames 
        frame_count=2

        if self.mode=='do3D':
            gray_stack = np.zeros(shape=(frame_count,image.shape[0],image.shape[1])) #Z,X,Y for cellpose 3D segmentation
        else:
            gray_stack = np.zeros(shape=(image.shape[0],image.shape[1],frame_count))

        calcium_stack = np.zeros(shape=(image.shape[0],image.shape[1],frame_count))

        frame_idx = 0
        while success:

            # TESTING: use only the first few frames
            if frame_idx == 2:
                break

            if self.mode=='do3D':
                gray_stack[frame_idx,:,:] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_stack[:,:,frame_idx] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            calcium_stack[:,:,frame_idx] = image[:,:,1]
            frame_idx+=1
            success,image = vidcap.read()

        return gray_stack, calcium_stack

    def process_data(self, cell_diameter, flow_thresh, cell_prob_thresh, resample, stitch_threshold=None):
        if self.mode=='stationary':
            self.process_stationary_data(cell_diameter, flow_thresh, cell_prob_thresh, resample)
        
        elif self.mode=='non-stationary':
            self.process_nonstationary_data(cell_diameter, flow_thresh, cell_prob_thresh, resample, stitch_threshold)

        elif self.mode=='do3D':
            self.process_3D_data(cell_diameter, flow_thresh, cell_prob_thresh, resample, stitch_threshold)
    
    def process_stationary_data(self, cell_diameter, flow_thresh, cell_prob_thresh, resample):
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
        # diameter = None #--> default diameter for 'cyto' model about somewhere around 20-30 pixels

        # Save out labeled image
        os.makedirs('results/', exist_ok=True)
        np.save('results/{}_segmentation.npy'.format(self.save_filename), masks)

        print(f'segmentation took {time.time()-start_time:.2f} seconds')

    
    def process_nonstationary_data(self, cell_diameter, flow_thresh, cell_prob_thresh, resample, stitch_threshold):
        start_time = time.time()

        os.makedirs('results/', exist_ok=True)

        # Image Preprocessing
        processed_gray_stack = np.copy(self.gray_stack)
        
        kernel_mc = np.ones((3,3),np.uint8) # morph closing kernel
        kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        for i in range(processed_gray_stack.shape[2]):
            # Enhance contrast
            processed_gray_stack[:,:,i] = cv2.normalize(processed_gray_stack[:,:,i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            # # Morphological closing
            processed_gray_stack[:,:,i] = cv2.morphologyEx(processed_gray_stack[:,:,i], cv2.MORPH_CLOSE, kernel_mc,iterations = 1)

            # TODO: add a background subtraction?

            # Sharpen the image
            processed_gray_stack[:,:,i] = cv2.filter2D(processed_gray_stack[:,:,i], -1, kernel_sharp)
            
            masks, flows, styles, dia = self.model.eval(processed_gray_stack[:,:,i], diameter=cell_diameter, channels=[0, 0], cellprob_threshold=cell_prob_thresh, flow_threshold=flow_thresh, resample=resample)
            np.save('results/{}_segmentation_{}.npy'.format(self.save_filename, i), masks)

        print(f'segmentation took {time.time()-start_time:.2f} seconds')

    def process_3D_data(self, cell_diameter, flow_thresh, cell_prob_thresh, resample, stitch_threshold):
        start_time = time.time()

        os.makedirs('results/', exist_ok=True)

        # Image Preprocessing
        processed_gray_stack = np.copy(self.gray_stack)
        
        kernel_mc = np.ones((3,3),np.uint8) # morph closing kernel
        kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        for i in range(processed_gray_stack.shape[0]): # iterate over frames
            # Enhance contrast
            processed_gray_stack[i,:,:] = cv2.normalize(processed_gray_stack[i,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            # # Morphological closing
            processed_gray_stack[i,:,:] = cv2.morphologyEx(processed_gray_stack[i,:,:], cv2.MORPH_CLOSE, kernel_mc,iterations = 1)

            # TODO: add a background subtraction?

            # Sharpen the image
            processed_gray_stack[i,:,:] = cv2.filter2D(processed_gray_stack[i,:,:], -1, kernel_sharp)
        
        masks_stitched, flows_stitched, styles_stitched, _ = self.model.eval(processed_gray_stack, channels=[0,0], diameter=cell_diameter, cellprob_threshold=cell_prob_thresh, flow_threshold=flow_thresh, resample=resample, do_3D=False, stitch_threshold=stitch_threshold)
        np.save('results/{}_segmentation.npy'.format(self.save_filename), masks_stitched)

        print(f'segmentation took {time.time()-start_time:.2f} seconds')

        return masks_stitched

    def visualize_segmentation(self):
        
        # Load segmentation
        if self.stationary:
            seg_path = 'results/'+self.filename.split('.mp4')[0]+'_segmentation.npy'
        else:
            seg_path = 'results/'+self.filename.split('.mp4')[0]+'_segmentation_0.npy' # first frame
        segmentation = np.load(seg_path, allow_pickle=True)
    
        # Load video
        vidcap = cv2.VideoCapture(self.filename)
        success,frame = vidcap.read() # Read first frame of video

        # Draw all contours 
        for i in range(1, len(np.unique(segmentation)) + 1):
            mask = segmentation == i
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            cv2.drawContours(frame, contours, 0, (255, 0, 0), 1)

        plt.imshow(frame)
        plt.axis('off')
        plt.show()

    def save_segmentation_video(self):

        # Load segmentation
        if self.stationary:
            seg_path = self.filename.split('.mp4')[0]+'_segmentation.npy'
            segmentation = np.load(seg_path, allow_pickle=True)
        else:
            seg_path = 'results/'+self.filename.split('.mp4')[0]+'_segmentation_{}.npy'
    
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
        os.makedirs('results/', exist_ok=True)
        out_contours = cv2.VideoWriter('results/'+self.save_filename+'_segmentation.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height))
        frame_idx = 0

        contour_images = []

        success,frame = vidcap.read()

        while(success):
            # Capture frame-by-frame

            if self.stationary:
                labels = segmentation
            if not self.stationary:
                labels = np.load(seg_path.format(frame_idx), allow_pickle=True)
            
            # Draw contours
            contour_image = np.copy(frame)
            for i in range(1, len(np.unique(labels)) + 1):
                mask = labels == i
                if np.sum(mask) > 0:
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)
            
            out_contours.write(contour_image)
            contour_images.append(contour_image)

            frame_idx += 1
            
            success, frame = vidcap.read()

        vidcap.release()
        out_contours.release()
        return contour_images, fps
    
    def extract_timeseries(self, save_centroids=True, save_pixels=False):

        if self.stationary:
            self.stationary_extraction(save_centroids, save_pixels)
        else:
            self.dynamic_extraction(save_centroids, save_pixels)

    def stationary_extraction(self, save_centroids, save_pixels):
        seg_path = self.filename.split('.mp4')[0]+'_segmentation.npy'
        segmentation = np.load(seg_path, allow_pickle=True)
        
        start_time = time.time()

        n_cells = np.max(segmentation) # max ROI ID
        n_timesteps = self.calcium_stack.shape[2]

        # Initialize empty time series matrix of shape # cells x # timesteps
        series = np.zeros(shape=(n_cells, n_timesteps))

        if save_centroids:
            # Initialize empty dataframe to store centroids for the indentified cells
            centroids_df = pd.DataFrame(columns=['label','series_index','x','y'])

        if save_pixels:
            # Initialize empty dataframe to store the label, x, and y of identified cells 
            pixels_df = pd.DataFrame(columns=['label','series_index','x','y'])

        # Image indices
        inds = np.indices((self.calcium_stack.shape[0], self.calcium_stack.shape[1]))

        # Loop through each frame
        for t in range(n_timesteps):
            # print(f'processing frame {t}')

            if self.stationary==False:
                labels = segmentation[t,:,:]
            else:
                labels = segmentation
            
            # Loop through each cell
            for label in range(1,n_cells+1): # first label=1 (0 is background)

                im = self.calcium_stack[:,:,t]
                series[label-1, t] = np.mean(im[labels==label])

                # Save out spatial data
                xs = inds[0,:,:][labels==label]
                ys = inds[1,:,:][labels==label]

                if save_centroids:
                    centroid_x = np.mean(xs)
                    centroid_y = np.mean(ys)

                    # Append to centroids_df
                    row = pd.Series({'label':int(label), 'series_index':int(label-1), 'x':centroid_x, 'y':centroid_y})

                    centroids_df = pd.concat([centroids_df, row.to_frame().T], ignore_index=True)
                    

                if save_pixels:
                    # Append to pixels_df 
                    for i in range(len(xs)):
                        rows = {'label':int(label), 'series_index':int(label-1), 'x':xs[i], 'y':ys[i]}
                        pixels_df = pixels_df.append(rows, ignore_index=True)

        print(series[:10,:10])

        print(f'extraction took {time.time()-start_time:.2f} seconds')
        np.savetxt(f'results/{self.save_filename}_series.csv', series, delimiter=',')
        if save_centroids:
            centroids_df.to_csv(f'results/{self.save_filename}_centroids.csv', sep=',',header=True, index=False)
        if save_pixels:
            pixels_df.to_csv(f'results/{self.save_filename}_pixels.csv', sep=',', header=True, index=False)

    def dynamic_extraction(self, save_centroids, save_pixels):
        pass

if __name__=='__main__':
    
    # Data
    FILENAME = '2022-10-13-6sLA-e 2-10s int+ATPimme.mp4' # bot_[id]_[phase]
    STATIONARY = 'False'

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

    seg = segModel(filename=FILENAME, stationary=STATIONARY)
    seg.process_data(cell_diameter=DIAMETER, flow_thresh=FLOW_THRESHOLD, cell_prob_thresh=CELL_PROB_THRESHOLD, resample=RESAMPLE, stitch_threshold=0.75)
