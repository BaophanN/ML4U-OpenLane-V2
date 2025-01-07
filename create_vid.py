import os
import cv2

def create_video_from_frames(dataset_folder, output_video_path, fps):
    # Get list of all folders in the dataset folder
    folders = []
    for folder in os.listdir(dataset_folder):
        temp_path = os.path.join(dataset_folder, folder) 
        for subfolder in os.listdir(temp_path): 
            name = os.path.join(temp_path,subfolder)
            folders.append(name)
    folders.sort()  # Ensure folders are processed in order

    # Initialize variables
    video_writer = None
    frame_size = None

    for folder in folders:
        # Construct paths to surround.jpg and bev.jpg
        surround_path = os.path.join(folder, 'surround.jpg')
        bev_path = os.path.join(folder, 'bev.jpg')
        
        # Check if both files exist
        if os.path.exists(surround_path) and os.path.exists(bev_path):
            # Read the images
            surround_img = cv2.imread(surround_path)
            bev_img = cv2.imread(bev_path)
            # print(surround_img.shape)
            # print(bev_img.shape)

            if surround_img is None or bev_img is None:
                print(f"Skipping folder {folder} due to invalid images.")
                continue
            
            # Ensure images have the same height for horizontal concatenation
            height = max(surround_img.shape[0], bev_img.shape[0]) # min(2843,807)
            width = max(surround_img.shape[1], bev_img.shape[1])
            surround_img_resized = cv2.resize(surround_img, (width,height))
            bev_img_resized = cv2.resize(bev_img, (width,height))
            # print("After reshape: ")
            # print(surround_img_resized.shape)
            # print(bev_img_resized.shape)
            
            
            # Concatenate images horizontally
            concatenated_frame = cv2.hconcat([surround_img_resized, bev_img_resized])
            
            if video_writer is None:
                # Initialize the video writer with the frame size
                frame_size = (concatenated_frame.shape[1], concatenated_frame.shape[0])
                video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
            
            # Write the concatenated frame to the video
            video_writer.write(concatenated_frame)
    
    # Release the video writer
    if video_writer is not None:
        video_writer.release()


if __name__ == "__main__":
    PATH = 'work_dirs/lanesegnet_r50_8x1_24e_olv2_subset_A/test/vis'
    output_video_path = 'video/lanesegnet_r50_8x1_24e_olv2_subset_A_vis.avi'
    create_video_from_frames(PATH, output_video_path, fps=1)


