#!bin/sh

# args: hyp_path, ref_path, ref_m2_path, save_dir, prefix

if test $# -ne 5
then
  echo "Bad arguments for $0"
  exit 1
fi

# errant_dir="/Users/you_pro/workspace/jnlp/bea2019/errant"
script_dir=$(cd $(dirname $0); pwd)
hypo_path="$1"
corr_path="$2"
ref_m2_path="$3"
save_dir="$script_dir/$4"
prefix="$5"
hyp_m2_path="$save_dir/$prefix.hyp.m2"
result_path="$save_dir/$prefix.result.txt"

# create M2 file: hyp.m2
python $ERRANT_DIR/parallel_to_m2.py -orig $hypo_path -cor $corr_path -out $hyp_m2_path

# compute ERRANT scores
python $ERRANT_DIR/compare_m2.py -hyp $hyp_m2_path -ref $ref_m2_path > $result_path

# extract an ERRANT score
sed -n 4p $result_path | cut -f 6
