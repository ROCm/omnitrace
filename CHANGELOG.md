# Changlog for Omnitrace

Full documentation for Omnitrace is available at
[https://rocm.docs.amd.com/projects/omnitrace/en/latest](https://rocm.docs.amd.com/projects/omnitrace/en/latest).

## Omnitrace v1.11.2 for rocm-6.2.0

### Changes

* Initial ROCm release

### Known Issues

* Loading output Perfetto files (the .proto file) in latest ui.perfetto.dev can
generate an error dialog box with the message:
"Oops, something went wrong!" and "Unknown field type".

  * As a workaround, the Perfetto files can be loaded with the previous version of the Perfetto UI:
    [https://ui.perfetto.dev/v46.0-35b3d9845/#!/](https://ui.perfetto.dev/v46.0-35b3d9845/#!/)
