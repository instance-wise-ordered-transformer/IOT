#!/usr/bin/env bash
if [ "$1" == '-h' ]
then
    echo "bash interactive_baseline.sh  cktpath --beam 5 --lenpen 1. -s en -t es "
fi
set -x
set -e
export PYTHONIOENCODING="UTF-8"

MOSE=$MOSE
sockeye=$sockeye

cktpath=$1
shift
if [ -z $cktpath ]
then
  exit
fi
numckts=$1
shift
cuda=$1
shift
export CUDA_VISIBLE_DEVICES=$cuda

POSITIONAL=()
beam=5
lenpen=1.0
srclng=en
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --beam)
    beam=$2; shift 2;;
  --lenpen)
    lenpen=$2; shift 2;;
  -s)
    srclng=$2; shift 2;;
  -t)
    tgtlng=$2; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done

if [ $tgtlng == 'es' ]; then
suffix="/data/iwslt/en_es/test.es.debpe.detok"
elif [ $srclng == 'es' ]; then
suffix='/data/iwslt/en_es/test.en.debpe.detok'
elif [ $tgtlng == 'zh' ]; then
suffix="-t iwslt17 -l ${srclng}-${tgtlng} --tok zh "
else
suffix="-t iwslt17 -l ${srclng}-${tgtlng}"
fi
if [ $srclng == 'en' ]
then
bpefile=/data/iwslt/${srclng}_${tgtlng}/test.$srclng
dictpath=/data/iwslt/${srclng}_${tgtlng}/databin
else
  bpefile=/data/iwslt/${tgtlng}_${srclng}/test.$srclng
dictpath=/data/iwslt/${tgtlng}_${srclng}/databin
fi
set -- "${POSITIONAL[@]}"
tgtlog=$cktpath.log

cat $bpefile  | python interactive.py $dictpath --path $cktpath --buffer-size 1024 \
--batch-size 128 --beam $beam --lenpen $lenpen -s $srclng -t $tgtlng --remove-bpe --num-ckts $numckts > $tgtlog
grep ^H $tgtlog  | cut -f3- > $tgtlog.h
if [ $tgtlng == 'zh' ]
then
    sed -r 's/ //g' $tgtlog.h > $tgtlog.h.detok
else
    perl $MOSE/scripts/tokenizer/detokenizer.perl -l $tgtlng < $tgtlog.h > $tgtlog.h.detok
fi
cat $tgtlog.h.detok | $sockeye/sockeye_contrib/sacrebleu/sacrebleu.py $suffix