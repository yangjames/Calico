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

Check out our [wiki pages](https://github.com/yangjames/Calico/wiki) for more info.

# License
Poor sensor calibration is a problem that is endemic within the robotics community, yet typically not given enough attention because it tends to detract from higher level project goals. The purpose of this library is to help roboticists quickly resolve their calibration issues so that they can work on more interesting things. This library is hereby granted the MIT license, to be used free of charge by a anyone within acadmia or industry.

To the best of my knowledge, all library dependencies also fall under BSD 2/3, LGPL, or Apache 2.0.

## MIT License

Copyright (c) 2023 James Yang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
