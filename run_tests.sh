#!/bin/bash
 
for i in 9 10 11 12 22 44 60 77 97 120 
do
	OUTFILE="00Random"$i
	mkdir $OUTFILE
	INFILE="formattedData/Random"$i".mc"
	
	python main.py $INFILE $OUTFILE 1 0 0

done

for i in 9 10 11 12 22 44 60 77 97 120
do
	OUTFILE="10Random"$i
	mkdir $OUTFILE
	INFILE="formattedData/Random"$i".mc"
	
	python main.py $INFILE $OUTFILE 1 1 0

done

for i in 9 10 11 12 22 44 60 77 97 120
do
	OUTFILE="11Random"$i
	mkdir $OUTFILE
	INFILE="formattedData/Random"$i".mc"
	
	python main.py $INFILE $OUTFILE 1 1 1

done

for i in 9 10 11 12 22 44 60 77 97 120
do
	OUTFILE="01Random"$i
	mkdir $OUTFILE
	INFILE="formattedData/Random"$i".mc"
	
	python main.py $INFILE $OUTFILE 1 0 1

done

for i in 9 10 11 12 
do
	OUTFILE="BFRandom"$i
	mkdir $OUTFILE
	INFILE="formattedData/Random"$i".mc"
	
	python main.py $INFILE $OUTFILE 1 0 1

done
