******************
What is Omnitrace?
******************

Omnitrace is an AMD open source research project and is not supported as part
of the ROCm software stack.

Omnitrace is designed for both high-level profiling and comprehensive tracing
of applications running on the CPU or the CPU+GPU via dynamic binary
instrumentation, call-stack sampling, and various other means for determining
currently executing function and line information.

Visualization of the comprehensive omnitrace results can be viewed in any modern
web browser by visiting [ui.perfetto.dev](https://ui.perfetto.dev/) and loading
the perfetto output (`.proto` files) produced by omnitrace.

Aggregated high-level results are available in text files for human consumption and JSON files for programmatic analysis.
The JSON output files are compatible with the python package [hatchet](https://github.com/hatchet/hatchet) which converts
the performance data into pandas dataframes and facilitate multi-run comparisons, filtering, visualization in Jupyter notebooks,
and much more.

 Omnitrace has two distinct configuration steps when instrumenting:

1. Configuring which functions and modules are instrumented in the target binaries (i.e. executable and/or libraries)
2. Configuring what the instrumentation does happens when the instrumented binaries are executed

