<p align="center">
<img src="https://user-images.githubusercontent.com/4121640/229179345-57bafb62-6391-498c-8d01-dbe86f8d54d1.png" width="300">
</p>

# Calico

Calico is a lightweight visual-inertial calibration library designed for rapid problem construction, debugging, and tool creation. Unlike other codebases that strictly contain standalone calibration tools made with specific hardware setups in mind, Calico is a flexible library with which you can build and change your **own tools** based on your hardware needs and limitations. As long as you adhere to the [geoemtry convention](https://github.com/yangjames/Calico/wiki/Geometry-Convention), there is **no need** to change the formulation of the underlying optimization.

Some features of Calico include:
- Sensor intrinsics, extrinsics, and latency estimation.
- Calibration with multiple (and mixed) fiducials.
- Addition of an arbitrary number of rigidly attached sensors.
- Measurement outlier tagging and exclusion.
- Robustifier kernels on a per-sensor basis.
- Addition of custom sensor intrinsics models.

Check out our [wiki pages](https://github.com/yangjames/Calico/wiki) for more info!
