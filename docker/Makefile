build: Dockerfile.build
	docker build -f Dockerfile.build . -t "oneflow-docker-build"

runtime: Dockerfile.runtime
	cp ../build/bin/oneflow ./bin/oneflow && docker build -f Dockerfile.build . -t "oneflow-docker-runtime"

build-test:
	docker run -it --rm --name build-test -v $(HOME)/src/oneflow:/opt/oneflow -v /opt/docker/mount:/mnt "oneflow-docker-build"

runtime-test:
	echo "runtime test"
