.PHONY: all build kernels run profile clean

all: build

build:
	bash scripts/build.sh

kernels:
	bash scripts/build_kernels.sh

run:
	bash scripts/run_all.sh

profile:
	@if [ -z "$(BENCH)" ]; then echo "usage: make profile BENCH=mte_copy_bw"; exit 2; fi
	bash scripts/profile.sh "$(BENCH)"

clean:
	rm -rf build results/*.csv results/*.txt results/summary.csv results/msprof_*
