#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export HYDRA_FULL_ERROR=1

CURR_AREA='9'  # set the area number accordingly [1,6]
CURR_DBSCAN=0.6
CURR_TOPK=-1
CURR_QUERY=100

python main_instance_segmentation.py \
  general.project_name="wheathead_iis" \
  general.experiment_name="wheathead_iis_${CURR_AREA}_from_scratch_2" \
  data.batch_size=4 \
  data/datasets=wheathead_iis \
  general.num_targets=4 \
  data.num_labels=3 \
  trainer.max_epochs=512 \
  general.area=${CURR_AREA} \
  trainer.val_check_interval=34

# python main_instance_segmentation.py \
#   general.experiment_name="wheathead_iis_${CURR_AREA}_from_scratch" \
#   general.checkpoint="saved/wheathead_iis_${CURR_AREA}_from_scratch/last.ckpt" \
#   general.train_mode=false \
#   general.area=${CURR_AREA} \
#   data.batch_size=4 \
#   data/datasets=wheathead_iis \
#   general.num_targets=4 \
#   data.num_labels=3 \
#   model.num_queries=${CURR_QUERY} \
#   general.topk_per_image=${CURR_TOPK} \
#   general.use_dbscan=true \
#   general.dbscan_eps=${CURR_DBSCAN} \
#   general.project_name="wheathead_iis"
