#!/bin/bash

if [ $# != 2 ]
then
   echo "usage: $0 <corpus> <order>"
   exit
fi 

kenlmpath=./kenlm/build
modelpath=./models
corpus=$1
order=$2

echo "Kenlm Path : $kenlmpath"
echo "Model Path : $modelpath"
echo "Corpus : $corpus"
echo "Order : $order"

# 0) decide which file to generate 
model="${corpus:3:${#corpus}}-${order}g"
arpa=$modelpath/$model.arpa 
binary=$modelpath/$model.bin 

# 1 ) computing an arpa model
echo "Training $order-gram KenLM model with data from $corpus and saving ARPA file to $arpa "
echo "" 
$kenlmpath/bin/lmplz -o $order -T /tmp < $corpus > $arpa
#$kenlmpath/bin/lmplz -o $order --interpolate_unigrams 0 -T /tmp < $corpus > $arpa

# 2) converting arpa into binary  
$kenlmpath/bin/build_binary $arpa $binary 

# 3) check
ls -l $arpa $binary
