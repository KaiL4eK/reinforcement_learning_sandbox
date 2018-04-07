#!/bin/bash

rsync -avzLPc --rsh='ssh -p9992' userquadro@uni1:~/neuroevol/neat/ff_last_pid .
rsync -avzLPc --rsh='ssh -p9992' userquadro@uni1:~/neuroevol/neat/ctrnn_last .
