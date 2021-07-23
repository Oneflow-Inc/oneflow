docker run -it --rm \
	-v /dataset:/dataset/ \
	oneflow-build \
    python3 -c "import oneflow"
