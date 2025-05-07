.PHONY: all openmp mpi seq

all: openmp mpi seq

openmp:
	$(MAKE) -C openmp all

mpi:
	$(MAKE) -C mpi all

seq:
	$(MAKE) -C seq all

clean:
	$(MAKE) -C openmp clean
	$(MAKE) -C mpi clean
	$(MAKE) -C seq clean

run:
	$(MAKE) -C openmp run
	$(MAKE) -C mpi run
	$(MAKE) -C seq run