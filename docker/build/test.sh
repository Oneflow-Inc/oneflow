docker run -it --rm \
	-v /dataset:/dataset/ \
	-v /home/caishenghang:/home/caishenghang \
	oneflow-build \
    python3 -c "import oneflow"
