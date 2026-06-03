#!/usr/bin/env zsh

if [ "$#" -ne 1 ]; then
   echo "Usage: $0 <model_dir>"
   exit 1
fi

set -e
model_dir="$1"

function csnotify {
   curl -d $1 ntfy.sh/W7T2QKNDH9Z4E3VJPRY8XACUL
}

# Generate event results
echo "Computing CV event validation results for model dir: $model_dir"
python nn_cmds.py compute-cv-event-val-results $model_dir 

# Generate event MERA results
echo "Running CV event MERA for model dir: $model_dir"
# python nn_cmds.py run-event-mera $model_dir --site-term --n-procs 10 && csnotify "CV event MERA complete" || csnotify "CV event MERA failed"
python nn_cmds.py run-event-mera $model_dir --site-term --ims "pSA_0.01" --ims "pSA_0.05" --ims "pSA_0.1" --ims "pSA_0.5" --ims "pSA_1.0" --ims "pSA_1.5" --ims "pSA_2.0" --ims "pSA_2.5" --ims "pSA_3.0" --ims "pSA_4.0" --ims "pSA_5.0" --ims "pSA_6.0" --ims "pSA_7.5" --ims "pSA_10.0" --n-procs 7 && csnotify "CV event MERA complete" || csnotify "CV event MERA failed"


# Generate MERA results 
# python nn_cmds.py run-mera $model_dir --site-term --n-procs 3 && csnotify "CV MERA run complete" || csnotify "CV MERA run failed"
python nn_cmds.py run-mera $model_dir --site-term --ims "pSA_0.01" --ims "pSA_0.05" --ims "pSA_0.1" --ims "pSA_0.5" --ims "pSA_1.0" --ims "pSA_1.5" --ims "pSA_2.0" --ims "pSA_2.5" --ims "pSA_3.0" --ims "pSA_4.0" --ims "pSA_5.0" --ims "pSA_6.0" --ims "pSA_7.5" --ims "pSA_10.0" --n-procs 3 && csnotify "CV MERA run complete" || csnotify "CV MERA run failed"


