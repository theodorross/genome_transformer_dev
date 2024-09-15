docker build --progress=plain -t theodorross/tensorflow-experiments:rocm-v0.3 -f dockerfile-rocm .
docker push theodorross/tensorflow-experiments:rocm-v0.3

docker build --progress=plain -t theodorross/tensorflow-experiments:cuda-v0.3 -f dockerfile-cuda .
docker push theodorross/tensorflow-experiments:cuda-v0.3