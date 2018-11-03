Bootstrap: docker
From: jcreinhold/synthnn

%runscript
exec echo "Try running with --app train/predict"

%apprun train
exec nn-train $@

%apprun predict
exec nn-predict $@