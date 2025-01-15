# Install virtual environment using conda
conda env create -f VAEs.yml
# Prepare data files according to dataloader tools
# Set up dataset/ model/ training configurations using config tools (could follow existing examples)
# Train autoencoders / VAEs 
Run train.py to start training
example command:
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config configs/${DATASET}/${experiment}.yaml >$logPath$logName 2>&1 &
# Extract latent features using the learned autoencoders / VAEs
Run project.py to project SNPs to latent variables
example command:
CUDA_VISIBLE_DEVICES=1 nohup python project.py --config configs/${DATASET}/${experiment}.yaml >$logPath$logName 2>&1 &
# Evaluate TSP performance (location/ age) using the extracted latent features
Run evaluate.py to test randomforest regressors
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --config configs/${DATASET}/${experiment}.yaml >$logPath$logName 2>&1 &


