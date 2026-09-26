# Token2Wav Optimization Report

The profiling-first branch does not assume that Token2Wav is the bottleneck.
Backend/kernel optimization can be selected only when backend wall/CUDA time
and an independent kernel or synchronization trace support the same conclusion.

No precision, CUDA Graph, fusion, remote backend, or Token2Wav serving
semantics are changed by the profiling instrumentation.
