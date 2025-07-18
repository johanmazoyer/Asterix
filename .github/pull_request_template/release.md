## Description

<!--
Give this PR a name following "Release vX.X.X" in the title above. You will want to pick an appropriate version
number depending on what the most recent release was, and on how much this release differs from the last one.
Provide a brief description of this release, and its circumstances, in this section.
-->

## Hardware and software tests

<!--
Perform all the tests in the table below and make sure the results are satisfactory. Put an x in the boxes ([ ]) of the
tests you ran successfully.
-->

|              | FDH, 2-DM control<br>600 modes | FDH, DM2 control<br>300 modes | HDH, DM2 control<br>300 modes |
|--------------|--------------------------------|-------------------------------|-------------------------------|
| **FQPM** sim |                                |                               |                               |
| **FQPM** hw  |                                |                               |                               |
| **WV** sim   |                                |                               |                               |
| **WV** hw    |                                |                               |                               |

FQPM ... four-quadrant phase-mask coronagraph  
WV ... wrapped vortex coronagraph  
hw ... hardware -> on the THD2 testbed  
sim ... simulations  
DM2 ... the in-pupil DM

✅ ... tested
🟡 ... untested

["converging EFC loop" specs TBD]  
[current idea: reaching a level of 1e-8 in less than 20 iterations and not diverging after 30 iterations]

Each box can be ticked if the corresponding matrix leads to a converging EFC loop, and on hardware without noticeable
DH degradations caused by the matrix or loop. FDH is a full dark hole, HDH is a half dark hole. The number of modes
indicate the number of modes you need to use in the matrix inversion when creating your dark hole for the tests.

The matrices created for the respective test cases need to be calculated without adjusting a single parameter in the
respective configfile, and after old simulation files have been deleted from disk - this includes matrix files
in the folder `Interaction_Matrices` as well as model files in `Model_local`.

## Matrix file names used

<!--
Please fill in the matrix file names you used for the tests above and save them on the RTC computer under
`F:\Control_matrices`
-->

FDH, 2-DM control: `20XX_please-fill-in`

FDH, DM2 control: `20XX_please-fill-in`

HDH, DM2 control: `20XX_please-fill-in`

## Preparation on hardware

1. Check the pupil alignments and actuators inside the pupil
2. Align science camera to focal-plane mask
3. Align tip-tilt

## Parameters used on hardware

Empty cells mean repeating values.

|                                | FDH, 2-DM control<br>600 modes | FDH, DM2 control<br>300 modes | HDH, DM2 control<br>300 modes |
|--------------------------------|--------------------------------|-------------------------------|-------------------------------|
| Exposure time science camera   |                                |                               |                               |
| Gain DM2 (in-pupil DM)         |                                |                               |                               |
| Gain DM1 (out-of-pupil DM)     |                                |                               |                               |
| Normalization data             |                                |                               |                               |
| Flux                           |                                |                               |                               |
| Lyot stop diameter             |                                |                               |                               |
| Pupil diameter                 |                                |                               |                               |


## Screenshots

<!--
Include DH images resulting from the EFC loops in the table above in this section. You can also add any other visuals you consider helpful.
-->

## Checklist

- [ ] This PR has the label "release" applied
- [ ] All boxes in the above table have been checked
- [ ] Images for each configuration have been added to the top-level comment in this PR, in the "Screenshots" section
- [ ] The used matrix for each hardware configuration has been saved to the matrix vault on RTC, under `F:\Control_matrices`
- [ ] The names of the used matrices have been put into the section "Matrix file names used"
