# TransformerRoofline Module

This directory contains the pre-compiled shared libraries (`.so` files) for the TransformerRoofline module. These libraries are essential for accurate performance modeling of transformer-based models in TokenSim.

## Platform Compatibility

The TransformerRoofline module is **only compatible with Linux systems**. This is because:
- The module is distributed as Linux shared libraries (`.so` files)
- These libraries are compiled specifically for Linux systems
- They are not compatible with macOS (`.dylib`) or Windows (`.dll`)

### Options for Non-Linux Users

If you're not running Linux, you have several options to use TransformerRoofline:

1. **Use a Linux Virtual Machine:**
   - Set up a Linux VM using VirtualBox, VMware, or similar
   - Install TokenSim and dependencies in the VM
   - Run your experiments within the VM

2. **Use Docker:**
   - Use our Docker container (if available) or create one
   - This provides a consistent Linux environment across platforms
   - Recommended for macOS and Windows users

3. **Use a Remote Linux Server:**
   - Run TokenSim on a remote Linux machine or cloud instance
   - Suitable for running large-scale experiments

4. **Skip TransformerRoofline Features:**
   - Run TokenSim without the performance modeling features
   - Some functionality will be limited

## Requirements

For Linux systems:
- Compatible with x86_64 or aarch64 architectures
- Required system libraries (check with `ldd`)
- Appropriate file permissions

## Directory Structure

The following files should be present in this directory:

```
TransformerRoofline/
├── __init__.py                     # Module initialization
├── roofline.so                     # Main shared library
└── README.md                       # This file
```

## Troubleshooting

If you encounter issues loading the shared libraries:

1. Verify that you're running on a Linux system
2. Check that the `.so` files are present in this directory
3. Ensure the libraries have executable permissions:
   ```bash
   chmod +x TransformerRoofline/*.so
   ```
4. Verify that your system architecture matches the compiled libraries
5. Check for any missing system dependencies using `ldd`

For any issues or questions, please refer to the main project documentation or open an issue on the project repository. 