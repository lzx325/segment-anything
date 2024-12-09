{
    set -e
    source "/home/liz0f/anaconda3/etc/profile.d/conda.sh"
    conda deactivate
    conda activate segment-anything
    python test.py
}