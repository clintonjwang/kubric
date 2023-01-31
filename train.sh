#! /bin/bash
cd /data/vision/polina/users/clintonw/code/kubric
source activate /data/vision/polina/users/clintonw/anaconda3/envs/nerfstudio
for i in $(seq $1 $2); do
ns-train nerfacto --data /data/vision/polina/scratch/clintonw/datasets/kubric/$i \
--experiment-name $i \
--pipeline.datamanager.camera-optimizer.mode off \
--viewer.quit-on-train-completion True
done
