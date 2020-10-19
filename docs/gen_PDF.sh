rm TeX/*
python3 gen_TeX.py
cd TeX/
for n_pool in 0 1 2 3
do
    pdflatex CNN_$n_pool.tex
done

