import cv2
import numpy as np
from cellpose import models, plot, utils
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import matplotlib as mpl
from skimage import measure


class segModel:
    def __init__(self, filename, save_filename=None, save_dir='results', gpu=True, model='cyto') -> None:
        self.model = models.Cellpose(gpu=gpu, model_type=model)
        self.filename=filename

        self.gray_stack, self.signal_stack, self.cell_membrane_stack, self.frame_count = self.get_frames()

        # keep copies of the original stacks so we can revert back
        self.gray_stack_original = self.gray_stack.copy()
        self.signal_stack_original = self.signal_stack.copy()
        self.cell_membrane_stack_original = self.cell_membrane_stack.copy()
        
        if save_filename is not None:
            self.save_filename = save_filename
        else:
            if len(filename.split('/'))>1:
                self.save_filename = self.filename.split('/')[-1].split('.mp4')[0]
            else:
                self.save_filename = self.filename.split('.mp4')[0]

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.masks = None

        self.colors=None

    def get_frames(self):

        # Extract frames from video
        vidcap = cv2.VideoCapture(self.filename)
        success,image = vidcap.read()

        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        gray_stack = np.zeros(shape=(image.shape[0],image.shape[1],frame_count))
        signal_stack = np.zeros(shape=(image.shape[0],image.shape[1],frame_count))
        cell_membrane_stack = np.zeros(shape=(image.shape[0],image.shape[1],frame_count))

        frame_idx = 0
        while success:

            gray_stack[:,:,frame_idx] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            signal_stack[:,:,frame_idx] = image[:,:,1] # green channel
            cell_membrane_stack[:,:,frame_idx] = image[:,:,2] # red channel

            frame_idx+=1
            success,image = vidcap.read()

        return gray_stack, signal_stack, cell_membrane_stack, frame_count
    
    def revert_back_to_original_video(self):
        self.gray_stack = self.gray_stack_original.copy()
        self.signal_stack = self.signal_stack_original.copy()
        self.cell_membrane_stack = self.cell_membrane_stack_original.copy()

    def trim_video(self, n_end=1):
        # n_end (int): number of frames to remove from the end of the video

        self.gray_stack = self.gray_stack[:,:,:-n_end]
        self.signal_stack = self.signal_stack[:,:,:-n_end]
        self.cell_membrane_stack = self.cell_membrane_stack[:,:,:-n_end]
        self.frame_count = self.gray_stack.shape[2] # update the frame count

    def remove_background(self, threshold=10):
        
        for i in range(self.frame_count):

            self.gray_stack[:,:,i] = self.keep_largest_cc(self.gray_stack[:,:,i], threshold)
            self.signal_stack[:,:,i] = self.keep_largest_cc(self.signal_stack[:,:,i], threshold)
            self.cell_membrane_stack[:,:,i] = self.keep_largest_cc(self.cell_membrane_stack[:,:,i], threshold)
        
    def keep_largest_cc(self, img, threshold):

        binary_image = img>threshold

        labeled_image, num_labels = measure.label(binary_image, background=0, return_num=True)

        if num_labels > 1:
            # Calculate the size of each connected component
            component_sizes = np.bincount(labeled_image.flat)[1:]

            # Find the label corresponding to the largest connected component
            largest_component_label = np.argmax(component_sizes) + 1
            
            # Create a mask to keep only the largest component
            largest_component_mask = (labeled_image == largest_component_label).astype(np.uint8)

            # Apply the mask to the original binary image
            largest_component_image = img * largest_component_mask
        else:
            # If there's only one component, return the original binary image
            largest_component_image = img

        return largest_component_image
        
    def visualize_frame(self, channel='gray', frame=0, use_projection=False, preprocess=False):

        if channel=='gray':
            img_stack = self.gray_stack.copy()
        elif channel=='signal':
            img_stack = self.signal_stack.copy()
        elif channel=='cell_membrane':
            img_stack = self.cell_membrane_stack.copy()

        if use_projection:
            img = np.mean(img_stack,axis=2)
        else:
            img = img_stack[:,:,frame]
        
        if preprocess:
            img = self.preprocess_image(img)

        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()

    def load_masks_from_npy(self, inpath):

        self.masks = np.load(inpath, allow_pickle=True).astype(int)
    
    def test_params(self, cell_diameter=30, flow_thresh=0.4, cell_prob_thresh=0.0, resample=False, preprocess=False, 
                    frame=0, channel='gray', mode='projection'):
        
        if channel=='gray':
            img_stack = self.gray_stack.copy()
        elif channel=='signal':
            img_stack = self.signal_stack.copy()
        elif channel=='cell_membrane':
            img_stack = self.cell_membrane_stack.copy()

        if mode=='projection':
            img = np.mean(img_stack,axis=2)
        else:
            img = img_stack[:,:,frame]
                
        if preprocess:
            img = self.preprocess_image(img)

        # Run model
        self.masks, flows, styles, diams = self.model.eval(img, diameter=cell_diameter, channels=[0, 0], cellprob_threshold=cell_prob_thresh, 
                                                         flow_threshold=flow_thresh, resample=resample)
        
        print(f'Diameter: {diams:.2f}')
        
        print('Num. segmented cells: ', len(np.unique(self.masks)))

        use_projection = True if mode=='projection' else False

        self.visualize_segmentation(n_frames=1, use_projection=use_projection, frame=frame, overlay_channel=channel)

        # Reset mask colors for when we actually run the model
        self.colors = None

    def preprocess_image(self, img):

        # Image Preprocessing

        # Enhance contrast
        # preprocessed_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Morphological closing
        kernel = np.ones((3,3),np.uint8)
        preprocessed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations = 1)

        # TODO: add a background subtraction?

        # Create the sharpening kernel
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # Sharpen the image
        # preprocessed_img = cv2.filter2D(img, -1, kernel)

        return preprocessed_img


    def run(self, cell_diameter=30, flow_thresh=0.4, cell_prob_thresh=0.0, resample=False, mode='projection', preprocess=False, channel='gray', stitch_threshold=0.001):

        if mode == '3D':
            # TODO: implement the cellpose 3D segmentation
            # Reshape the image stack to (Z,X,Y)
            # Run the segmentation
            # Reshape the mask to (X,Y,Z)
            print('Cellpose 3D segmentation not yet implemented.')
            return

        self.masks = self.process_data_2D(cell_diameter, flow_thresh, cell_prob_thresh, resample, mode, preprocess, channel, stitch_threshold)

    def process_data_2D(self, cell_diameter, flow_thresh, cell_prob_thresh, resample, mode, preprocess, channel, stitch_threshold):
        start_time = time.time()

        if channel=='gray':
            img_stack = self.gray_stack.copy()
        elif channel=='signal':
            img_stack = self.signal_stack.copy()
        elif channel=='cell_membrane':
            img_stack = self.cell_membrane_stack.copy()

            img = np.mean(img_stack,axis=2)

        if mode == 'projection':

            img = np.mean(img_stack,axis=2)

            if preprocess:
                img = self.preprocess_image(img)

            masks, flows, styles, dia = self.model.eval(img, diameter=cell_diameter, channels=[0, 0], cellprob_threshold=cell_prob_thresh, flow_threshold=flow_thresh, resample=resample)

            # Save out labeled image
            np.save('{}/{}_projection_segmentation.npy'.format(self.save_dir,self.save_filename), masks)

        elif mode=='stitch':

            # Initialize empty array to store masks
            masks = np.zeros(shape=img_stack.shape)

            curr_mask, flows, styles, dia = self.model.eval(img_stack[:,:,0], diameter=cell_diameter, channels=[0, 0], cellprob_threshold=cell_prob_thresh, flow_threshold=flow_thresh, resample=resample)
            masks[:,:,0] = curr_mask
            
            # Iterate through all the frames
            for i in range(1,img_stack.shape[2]):
                next_mask, flows, styles, dia = self.model.eval(img_stack[:,:,i], diameter=cell_diameter, channels=[0, 0], cellprob_threshold=cell_prob_thresh, flow_threshold=flow_thresh, resample=resample)
                
                masks_stitched = utils.stitch3D([curr_mask,next_mask], stitch_threshold=stitch_threshold)
                
                masks[:,:,i] = masks_stitched[1]

                curr_mask = masks_stitched[1]

            print('Cleaning masks...')
            masks = self.clean_segmentation_3D(masks)
            
            # Save out labeled image
            np.save('{}/{}_stitched_segmentation.npy'.format(self.save_dir,self.save_filename), masks)

        else:
            print("Mode must be 'projection' or 'stitch'.")
            return 

        print(f'Segmentation took {time.time()-start_time:.2f} seconds.')
        return masks


    # def process_data_3D(self, cell_diameter, flow_thresh, cell_prob_thresh, resample, stitch_threshold):
    #     start_time = time.time()

    #     os.makedirs('results/', exist_ok=True)
        
    #     masks_stitched, flows_stitched, styles_stitched, _ = self.model.eval(processed_gray_stack, channels=[0,0], diameter=cell_diameter, cellprob_threshold=cell_prob_thresh, flow_threshold=flow_thresh, resample=resample, do_3D=False, stitch_threshold=stitch_threshold)
    #     np.save('results/{}_segmentation.npy'.format(self.save_filename), masks_stitched)

    #     print(f'segmentation took {time.time()-start_time:.2f} seconds')

    #     return masks_stitched
        
    # def process_3D_data(self, cell_diameter, flow_thresh, cell_prob_thresh, resample, stitch_threshold, segment_every=3):
    #     start_time = time.time()

    #     os.makedirs('results/', exist_ok=True)

    #     # Run first segmentation
    #     curr_masks, _, _, _ = self.model.eval(processed_gray_stack[0:segment_every], channels=[0,0], diameter=cell_diameter, cellprob_threshold=cell_prob_thresh, flow_threshold=flow_thresh, resample=resample, do_3D=False, stitch_threshold=stitch_threshold)
        
    #     # Iterate through every segment_every
    #     for j in range(1,processed_gray_stack.shape[0]//segment_every+1):
    #         next_masks, _, _, _ = self.model.eval(processed_gray_stack[(segment_every-1)*j:(segment_every-1)*j+segment_every], channels=[0,0], diameter=cell_diameter, cellprob_threshold=cell_prob_thresh, flow_threshold=flow_thresh, resample=resample, do_3D=False, stitch_threshold=stitch_threshold)

    #         # TODO: combine curr_masks and next_masks 

    #         return curr_masks, next_masks
        
        # masks_stitched, flows_stitched, styles_stitched, _ = self.model.eval(processed_gray_stack, channels=[0,0], diameter=cell_diameter, cellprob_threshold=cell_prob_thresh, flow_threshold=flow_thresh, resample=resample, do_3D=False, stitch_threshold=stitch_threshold)
        # np.save('results/{}_segmentation.npy'.format(self.save_filename), masks_stitched)

        # print(f'segmentation took {time.time()-start_time:.2f} seconds')

        # return masks_stitched
    

    def clean_segmentation_3D(self, masks):

        if len(masks.shape)<3:
            print('3D mask required.')
            return

        # Keep only labels that exist in all frames

        # Determine which labels 
        set1 = set(masks[:,:,0].flatten())

        differences = []

        for z in range(1, masks.shape[2]):

            set2 = set(masks[:,:,z].flatten())

            intersection = set1.intersection(set2)
            difference = (set1.union(set2)) - intersection

            differences+=list(difference)

            set1 = intersection 

        print('Num. consistent labels:', len(intersection))
        print('Num. erratic labels:', len(differences))

        # Clean segmentation

        # remove IDs in difference
        masks[np.isin(masks, differences)] = 0

        # for j in list(differences):
        #     try:
        #         masks[masks==j] = 0 # remove IDs in difference
        #     except:
        #         continue

        # Reset labels 
        # Create a mask for the intersection values
        intersection_mask = np.isin(masks, intersection)

        # Replace intersection values with consecutive labels
        unique_values, inverse = np.unique(masks[intersection_mask], return_inverse=True)
        labels = np.arange(1, len(unique_values) + 1)
        masks[intersection_mask] = labels[inverse] 

        # label_count = 1

        # # should be able to do this more quickly
        # for i,k in enumerate(list(intersection)):

        #     if k==0:
        #         continue

        #     masks[masks==k] = label_count

        #     label_count+=1

        return masks

    def set_label_colors(self):
        # Generate list of unique colors for each segmentation
        cmap = mpl.colormaps['hsv']
        self.colors = cmap(np.random.random(len(np.unique(self.masks))))[:,:-1] # get rid of alpha 

    def visualize_segmentation(self, n_frames=1, vis_type='fill', overlay=False, overlay_channel='gray', use_projection=False, frame=0): 
        '''
        Visualize segmentation

        n = number of frames to plot with segmentation overlayed (only works with use_projection=False)
        vis_type = 'fill' or 'outline' (default='fill'). Whether to fill the identified cells or just outline them.
        overlay = Bool. (default=False). Overlay segmentations over original image or use a white background for clarity. 
        overlay_channel = if overlay=True, this parameter specifies the image channel on which the segmentation is overlaid. 
        use_projection = compute the image projection in the appropriate channel (specified by overlay_channel).
        frame = which frame to plot if n_frames=1 and use_projection=False.
        
        # TODO: implement vis_type=='outline'
        '''

        if overlay_channel=='gray':
            img_stack = self.gray_stack.copy()
        elif overlay_channel=='signal':
            img_stack = self.signal_stack.copy()
        elif overlay_channel=='cell_membrane':
            img_stack = self.cell_membrane_stack.copy()

        if use_projection:
            n_frames=1

        if self.colors is None:
            self.set_label_colors()

        if n_frames==1:

            if use_projection:
                imgout = np.mean(img_stack,axis=2)
            else:
                imgout = img_stack[:,:,frame]

            if len(self.masks.shape)==2:
                seg = self.masks.copy()
            else:
                seg = self.masks[:,:,frame].copy()

            imgout = plot.mask_overlay(imgout, seg, self.colors)

            plt.imshow(imgout)
            plt.axis('off')

            if use_projection:
                plt.title('projection')
            else:
                plt.title(f't={frame}')

            plt.show()


        else:
            # Plot multiple frame
            fig, axes = plt.subplots(1, n_frames)

            for t in range(n_frames):
                
                ax = axes[t]

                if overlay:
                    imgout = img_stack[:,:,t].copy()
                else:
                    imgout = np.ones(shape=img_stack[:,:,t].shape) # white background

                if len(self.masks.shape)==2:
                    seg = self.masks.copy()
                else:
                    seg = self.masks[:,:,t].copy()

                imgout = plot.mask_overlay(imgout, seg, self.colors)

                ax.imshow(imgout)
                ax.axis('off')
                ax.set_title(f't={t}')

            plt.show()
    
    def save_segmentation_video(self, overlay_channel='gray', save_path=None, vis_type='fill', overlay=True,codec='mp4v'):
        # codec = 'mp4v' or 'MJPG'

        if overlay_channel=='gray':
            img_stack = self.gray_stack.copy()
        elif overlay_channel=='signal':
            img_stack = self.signal_stack.copy()
        elif overlay_channel=='cell_membrane':
            img_stack = self.cell_membrane_stack.copy()

        if self.colors is None:
            self.set_label_colors()

        # Read video
        vidcap = cv2.VideoCapture(self.filename)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # get fps from video
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print(f'{frame_count} frames in video')

        # Get frame width and height
        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))

        vidcap.release()

        if save_path is not None:
            out_video = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*codec), fps, (frame_width,frame_height))
        else:
            out_video = cv2.VideoWriter(f'{self.save_dir}/{self.save_filename}_segmentation_video.mp4',cv2.VideoWriter_fourcc(*codec), fps, (frame_width,frame_height))

        frame_idx = 0

        if len(self.masks.shape)==3:
            n_frames = np.min([img_stack.shape[2],self.masks.shape[2]])
        else:
            n_frames = img_stack.shape[2]
            seg = self.masks

        for i,t in enumerate(range(n_frames)):
        
            if overlay:
                imgout= img_stack[:, :, t].copy()
            else:
                imgout=np.ones(shape=img_stack[:,:,t].shape)
            
            if len(self.masks.shape)==3:
                seg = self.masks[:,:,t]
        
            imgout = plot.mask_overlay(imgout, seg, self.colors)

            out_video.write(imgout)
            
            frame_idx += 1

        out_video.release()
    
    def extract_timeseries(self, save_centroids=True, save_pixels=False):

        if self.masks is None:
            print('Run model to generate masks or load from .npy file,')
            return
        
        static_segmentation=True if len(self.masks.shape)==2 else False

        start_time = time.time()

        n_cells = np.max(self.masks) # max ROI ID

        if len(self.masks.shape)==3:
            n_timesteps = np.min([self.signal_stack.shape[2],self.masks.shape[2]])
        else:
            n_timesteps = self.signal_stack.shape[2]

        # Initialize empty time series matrix of shape # cells x # timesteps
        series = np.zeros(shape=(n_cells, n_timesteps))

        if save_centroids:
            # Initialize empty dataframe to store centroids for the indentified cells
            centroids_df = pd.DataFrame(columns=['label','series_index','x','y'])

        if save_pixels:
            # Initialize empty dataframe to store the label, x, and y of identified cells 
            pixels_df = pd.DataFrame(columns=['label','series_index','x','y'])

        # Image indices
        inds = np.indices((self.signal_stack.shape[0], self.signal_stack.shape[1]))

        # Loop through each frame
        for t in range(n_timesteps):
            # print(f'processing frame {t}')

            if static_segmentation==False:
                labels = self.masks[:,:,t]
            else:
                labels = self.masks
            
            # Loop through each cell
            for label in range(1,n_cells+1): # first label=1 (0 is background)

                im = self.signal_stack[:,:,t]
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
        np.savetxt(f'{self.save_dir}/{self.save_filename}_series.csv', series, delimiter=',')
        if save_centroids:
            centroids_df.to_csv(f'{self.save_dir}/{self.save_filename}_centroids.csv', sep=',',header=True, index=False)
        if save_pixels:
            pixels_df.to_csv(f'{self.save_dir}/{self.save_filename}_pixels.csv', sep=',', header=True, index=False)

    # def visualize_segmentation(self):
        
    #     # Load segmentation
    #     if self.stationary:
    #         seg_path = 'results/'+self.filename.split('.mp4')[0]+'_segmentation.npy'
    #     else:
    #         seg_path = 'results/'+self.filename.split('.mp4')[0]+'_segmentation_0.npy' # first frame
    #     segmentation = np.load(seg_path, allow_pickle=True)
    
    #     # Load video
    #     vidcap = cv2.VideoCapture(self.filename)
    #     success,frame = vidcap.read() # Read first frame of video

    #     # Draw all contours 
    #     for i in range(1, len(np.unique(segmentation)) + 1):
    #         mask = segmentation == i
    #         contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #         cv2.drawContours(frame, contours, 0, (255, 0, 0), 1)

    #     plt.imshow(frame)
    #     plt.axis('off')
    #     plt.show()

    # def save_segmentation_video(self):

    #     # Load segmentation
    #     if self.stationary:
    #         seg_path = self.filename.split('.mp4')[0]+'_segmentation.npy'
    #         segmentation = np.load(seg_path, allow_pickle=True)
    #     else:
    #         seg_path = 'results/'+self.filename.split('.mp4')[0]+'_segmentation_{}.npy'
    
    #     # Read video
    #     vidcap = cv2.VideoCapture(self.filename)
    #     frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    #     # get fps from video
    #     fps = vidcap.get(cv2.CAP_PROP_FPS)
    #     print(f'{frame_count} frames in video')

    #     # Get frame width and height
    #     frame_width = int(vidcap.get(3))
    #     frame_height = int(vidcap.get(4))

    #     # Define the codec and create VideoWriter object in .mp4 format
    #     os.makedirs('results/', exist_ok=True)
    #     out_contours = cv2.VideoWriter('results/'+self.save_filename+'_segmentation.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height))
    #     frame_idx = 0

    #     contour_images = []

    #     success,frame = vidcap.read()

    #     while(success):
    #         # Capture frame-by-frame

    #         if self.stationary:
    #             labels = segmentation
    #         if not self.stationary:
    #             labels = np.load(seg_path.format(frame_idx), allow_pickle=True)
            
    #         # Draw contours
    #         contour_image = np.copy(frame)
    #         for i in range(1, len(np.unique(labels)) + 1):
    #             mask = labels == i
    #             if np.sum(mask) > 0:
    #                 contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #                 cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)
            
    #         out_contours.write(contour_image)
    #         contour_images.append(contour_image)

    #         frame_idx += 1
            
    #         success, frame = vidcap.read()

    #     vidcap.release()
    #     out_contours.release()
    #     return contour_images, fps
            

    # Another way to do outlines... not sure if I need

    # # Plot single mask
    # plt.figure(figsize=(5,3))

    # maskID = 10
    # single_ID_mask1 = masks[0].copy()
    # single_ID_mask1[single_ID_mask1!=maskID] = 0 # remove all IDs that are not the one of interest

    # single_ID_mask2 = masks[1].copy()
    # single_ID_mask2[single_ID_mask2!=maskID] = 0 # remove all IDs that are not the one of interest

    # single_ID_masks = [single_ID_mask1, single_ID_mask2]
    # iplanes = [2,3]

    # for i,iplane in enumerate(np.arange(2)):

    # plt.subplot(1,2,i+1)

    # outlines = utils.masks_to_outlines(single_ID_masks[i])
    # outX, outY = np.nonzero(outlines)

    # imgout= model.gray_stack[iplanes[i], :, :].copy()

    # imgout[outX, outY] = 255

    # # imgout = plot.mask_overlay(model.gray_stack[iplane, :, :].copy(), single_ID_masks[i])

    # plt.imshow(imgout)
    # plt.title('iplane = %d'%iplanes[i])