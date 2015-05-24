
---

## Latest news ##

  * **[2015-01-27]** I will be attending [SPIE Medical Imaging](http://spie.org/medical-imaging.xml) conference in Orlando at the end of February. If you are a user of MC-GPU and interested in talking about the last developments on the code, just let me know!

  * **[2015-01-22]** We have developed a method to modify the PENELOPE 2006 material files to use experimentally measured molecular form factors instead of theoretical form factors computed with the independent atom approximation. The material files include Molecular Interference Function (MIF) correction and can be used to improve the realism of the small-angle coherent x-ray scattering simulated with the MC-GPU and PENELOPE codes. Information on how to implement the MIF extension and an example simulation is available at the website of the [MC-GPU-MIF\_extension project](http://code.google.com/p/mcgpu-mif-extension/).

  * **[2012-12-12]** MC-GPU v.1.3 upgraded to CUDA 5 available in the "Download" tab!!!


The Doxygen documentation for the new version of MC-GPU is available [here](http://mcgpu.googlecode.com/hg/wiki/MC-GPU_v1.3_DOXYGEN_code_reference.html).
Check the ["Source"](http://code.google.com/p/mcgpu/source/list) tab for possible bug corrections in the released code.
Please use the ["Issues"](http://code.google.com/p/mcgpu/issues/list) tab to post any comment/correction/question about the code. Enjoy the simulations!


---


# Project summary #

**MC-GPU** is a GPU-accelerated x-ray transport simulation code that can generate clinically-realistic radiographic projection images and computed tomography (CT) scans of the human anatomy.
As it is explained in the [#Disclaimer](#Disclaimer.md) section below, this code is in the public domain and it can be used and distributed for free.
The HTML documentation of MC-GPU is available [here](http://mcgpu.googlecode.com/hg/wiki/index.html).

MC-GPU implements a massively multi-threaded Monte Carlo simulation algorithm for the transport of x rays in a voxelized geometry and uses the x-ray interaction models and cross sections from [PENELOPE 2006](http://www.oecd-nea.org/tools/abstract/detail/nea-1525). The code can handle realistic human anatomy phantoms, for example the freely available models from the [Virtual Family](http://www.itis.ethz.ch/services/human-and-animal-models/human-models/). Electron transport is not implemented. The code has been developed using the [CUDA](http://www.nvidia.com/object/cuda_what_is.html) programming model and the simulation can be executed in parallel in state-of-the-art GPUs from NVIDIA Corporation (Santa Clara, CA, USA). An MPI library is used to address multiple GPUs in parallel during the CT simulations. In typical diagnostic imaging simulations, a 15 to 30-fold speed up is obtained using a GPU compared to a CPU execution.

MC-GPU is being developed at the _U. S. Food and Drug Administration (FDA), Center for Devices and Radiological Health, Office of Science and Engineering Laboratories_, [Division of Imaging and Applied Mathematics](http://www.fda.gov/AboutFDA/CentersOffices/OfficeofMedicalProductsandTobacco/CDRH/CDRHOffices/ucm299950.htm). Partial funding for the development of this software has been provided by the FDA _Office of the Chief Scientist_ through the Computational Endpoints for Cardiovascular Device Evaluations project. This code is still in development, please report to the authors any issue/bug that you may encounter. Feel free to use the Wiki section to suggest improvements to the code too.

The MC-GPU code was first introduced in the paper listed below, which should be referenced by researchers using this code. The software is also described in chapter 50 of the book [GPU Computing Gems](http://mkp.com/gpu-computing-gems) (Emerald Edition) edited by Wen-mei W. Hwu.

  * _Andreu Badal and Aldo Badano, "Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel Graphics Processing Unit", Medical Physics 36, pp. 4878–4880 (2009)._


---


## Sample image ##

Sample projection radiography image of the [Duke phantom](http://www.itis.ethz.ch/services/human-and-animal-models/human-models/duke/) simulated with MC-GPU.
This image was generated tracking 10<sup>10</sup>, 50 keV x-rays. The simulation required 112 minutes of computation in a NVIDIA Tesla C1060 GPU (simulation speed: 1487676.8 x-rays/second).

> ![http://mcgpu.googlecode.com/files/mc-gpu_1mmDuke_50keV_1e10hist__All_and_NoScatter_LowRes.png](http://mcgpu.googlecode.com/files/mc-gpu_1mmDuke_50keV_1e10hist__All_and_NoScatter_LowRes.png)


The image on the left includes both the primary and scattered x-rays, while the one on the right contains only primary particles (scatter-free). The gray scale has units of eV/cm<sup>2</sup> per x ray. Therefore darker pixels have larger signal, meaning that the primary x rays encountered less attenuating materials traveling from the x-ray source to this pixel or that more scattered radiation was detected in the pixel.




---


## Disclaimer ##

This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to [Title 17, Section 105 of the United States Code](http://www.copyright.gov/title17/92chap1.html#105), this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic.   Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions.  Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.