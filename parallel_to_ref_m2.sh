#!bin/sh

# args: orig_path, ref_path, save_path
if test $# -ne 3
then
  echo "Bad arguments for $0"
  exit 1
fi
# create M2 file: ref.m2
python $ERRANT_DIR/parallel_to_m2.py -orig $1 -cor $2 -out $3
