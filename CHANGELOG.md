# Changlog for Omnitrace

Full documentation for Omnitrace is available at
[https://rocm.docs.amd.com/projects/omnitrace/en/latest](https://rocm.docs.amd.com/projects/omnitrace/en/latest).

## Omnitrace v1.11.2 for rocm-6.2.0

### Known Issues

* Perfetto can no longer open Omnitrace proto files.
Loading the Perfetto trace output `.proto` file in `ui.perfetto.dev` can result
in a dialog with the message, "Oops, something went wrong! Please file a bug."
The information in the dialog will refer to an "Unknown field type." The workaround
is to open the files with the previous version of the Perfetto UI found at
[https://ui.perfetto.dev/v46.0-35b3d9845/#!/](https://ui.perfetto.dev/v46.0-35b3d9845/#!/).
