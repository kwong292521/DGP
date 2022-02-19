# DGP
An python implementation of the paper [Surface-from-Gradients: An Approach Based on Discrete Geometry Processing](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Xie_Surface-from-Gradients_An_Approach_2014_CVPR_paper.html)

This method aims at reconstruct the surface from the normal map compute by Photometric Stereo.

In python implementation, cholesky factorization in sparse matrix could not excuted(The implementation the paper mentioned is generated on C++ with the help of TAUCS Library, and I will try to implement the C++ version in the future)

I wrote a comprehension of the paper [here](https://blog.csdn.net/SZU_Kwong/article/details/123013606)

## Run the code
Clone the repository and run the DGP.py

if you use your own data
```
python DGP.py -n [path-to-normal-file] -m [path-to-mask-file] -o [path-to-output-obj-file] -i 1
```
the default is the bunny obj

if you want to test the obj from Diligent dataset, such as cat
```
python DPG.py --obj cat_mat_png -i 1
```