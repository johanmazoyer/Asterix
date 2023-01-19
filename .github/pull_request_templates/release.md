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

|          | FDH, 2DM control<br>600 modes | FDH, 1DM control<br>300 modes | HDH, 1DM control<br>300 modes |
|----------|-------------------------------|-------------------------------|-------------------------------|
| **FQPM** sim | [ ]                           | [ ]                           | [ ]                           |
| **FQPM** hw  | [ ]                           | [ ]                           | [ ]                           |
| **WV** sim   | [ ]                           | [ ]                           | [ ]                           |
| **WV** hw    | [ ]                           | [ ]                           | [ ]                           |

FQPM ... four-quadrant phase-mask coronagraph
WV ... wrapped vortex coronagraph
hw ... hardware -> on the THD2 testbed
sim ... simulations

Each box can be ticked if the corresponding matrix leads to a converging EFC loop, and on hardware without noticeable
DH degradations caused by the matrix or loop. FDH is a full dark hole, HDH is a half dark hole. The number of modes
indicate the number of modes you need to use in the matrix inversion when creating your matrices for the tests.

The matrices created for the respective test cases need to be calculated without adjusting a single parameter in the
respective configfile, and after old simulation files have been deleted from disk - this includes matrix files
in the folder `Interaction_Matrices` as well as model files in `Model_local`.

## Screenshots

<!--
Include DH images resulting from the EFC loops in the table above in this section. You can also add any other visuals you consider helpful.
-->

## Checklist

- [ ] This PR has the label "release" applied
- [ ] All boxes in the above table have been checked
- [ ] Images for each configuration have been added to the top-level comment in this PR, in the "Screenshots" section
- [ ] The used matrix for each hardware configuration has been saved to the matrix vault on RTC, under `F:\Control_matrices`
