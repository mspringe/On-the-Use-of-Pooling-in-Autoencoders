#!/bin/bash
for n_pool in 0 1 2 3
do
    convert           \
       -verbose       \
       -density 150   \
       -trim          \
        TeX/CNN_$n_pool.pdf      \
       -quality 100   \
        figures/CNN_$n_pool.png
done
