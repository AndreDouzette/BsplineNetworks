TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared spline.cc -o ./ubuntu/spline.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared splineGrad.cc -o ./ubuntu/splineGrad.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared sortKnot.cc -o ./ubuntu/sortKnot.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared sortKnotGrad.cc -o ./ubuntu/sortKnotGrad.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared fixedSpline.cc -o ./ubuntu/fixedSpline.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared fixedSplineGrad.cc -o ./ubuntu/fixedSplineGrad.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0