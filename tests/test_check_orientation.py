import numpy as np

from tools.check_orientation import projection_panels


def test_projection_panels_put_superior_and_anterior_on_top():
    volume = np.full((10, 20, 30), -1000.0, dtype=np.float32)   # R x A x S
    volume[5, 10, 28] = 2000.0      # near superior end
    volume[5, 18, 5] = 2000.0       # near anterior end
    coronal, sagittal, axial = projection_panels(volume)
    assert coronal.shape == (30, 10) and sagittal.shape == (30, 20) and axial.shape == (20, 10)
    assert coronal[:5].max() == 255          # superior voxel lands in the top rows
    assert sagittal[:5].max() == 255
    assert axial[:5].max() == 255            # anterior voxel lands in the top rows of the axial view
    assert axial[-5:].max() < 255
