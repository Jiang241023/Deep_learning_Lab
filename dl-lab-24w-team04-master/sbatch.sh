#!/bin/bash -l
 
# Slurm parameters
#SBATCH --job-name=job_name
#SBATCH --output=job_name-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
 
# Activate everything you need
module load cuda/11.8
# Run your python code

# First project
# 1.Run the resize.py
python DL_LAB_Diabetic_Retinopathy_Detection/diabetic_retinopathy/input_pipeline/resize.py

# 2.Run the dataclass.py
#python DL_LAB_Diabetic_Retinopathy_Detection/diabetic_retinopathy/input_pipeline/dataclass.py

# 3.Run the data_balance.py
#python DL_LAB_Diabetic_Retinopathy_Detection/diabetic_retinopathy/input_pipeline/data_balance.py

# 4.Run the main.py file with FLAGS.train = True
#python DL_LAB_Diabetic_Retinopathy_Detection/diabetic_retinopathy/main.py

# 5.Run the main.py file with FLAGS.train = False
#python DL_LAB_Diabetic_Retinopathy_Detection/diabetic_retinopathy/main.py

# 6.Update ensemble = True in main.py for ensembled results.
#python /home/RUS_CIP/st186731/dl-lab-24w-team04/diabetic_retinopathy/main.py

# 7.Run the wandb_sweep.py (Please add comments to train_model.unfrz_layer and grad_cam_visualization.img_path, before running this file or it will raise errors)
#python /home/RUS_CIP/st186731/dl-lab-24w-team04/diabetic_retinopathy/wandb_sweep.py

# 8. Run the GRAD_CAM_visualization.py
#python /home/RUS_CIP/st186731/dl-lab-24w-team04/diabetic_retinopathy/deep_visualization/GRAD_CAM_visualization.py

# Second project
# 1.Run the main.py with FLAGS.train = True for gru_like 
#python DL_LAB_HAPT/HAR/Human_Activity_Recognition/main.py

# 2.Run the main.py with FLAGS.train = True for lstm_like (please according to the readme to adjust some codes to run it)
#python DL_LAB_HAPT/HAR/Human_Activity_Recognition/main.py

# 3.Run the main.py with  FLAGS.train = False for gru_like 
#python DL_LAB_HAPT/HAR/Human_Activity_Recognition/main.py

# 4.Run the main.py with FLAGS.train = False for lstm_like (please according to the readme to adjust some codes to run it)
#python DL_LAB_HAPT/HAR/Human_Activity_Recognition/main.py

# 5.Run the visualization.py 
#python DL_LAB_HAPT/HAR/Human_Activity_Recognition/visualization/visualization.py

# 6.Run the wandb_sweep.py (Please accordng to the instructions of readme)
#python DL_LAB_HAPT/HAR/Human_Activity_Recognition/wandb_sweep.py
