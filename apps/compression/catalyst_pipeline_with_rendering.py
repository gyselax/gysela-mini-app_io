# script-version: 2.0
from paraview.simple import *
from paraview import catalyst
import time

paraview.simple._DisableFirstRenderCameraReset()
# registrationName must match the channel name used in the
# 'CatalystAdaptor'.
producer = TrivialProducer(registrationName="grid")

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
# renderView1.UseOffscreenRendering = 1
renderView1.MultiSamples = 0
renderView1.DepthPeeling = 0
renderView1.StereoRender = 0
renderView1.EnableRayTracing = 0

renderView1.ViewSize = [1600,800]
renderView1.CameraPosition = [157.90070691620653, 64.91180236667495, 167.90421495515105]
renderView1.CameraFocalPoint = [19.452526958533134, 28.491610229010647, 10.883993417012459]
renderView1.CameraViewUp = [0.07934883419275315, 0.953396338566962, -0.2910999555468221]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 54.99504523136608
renderView1.UseColorPaletteForBackground = 0
renderView1.Background = [0.0, 0.0, 0.0]

# get color transfer function/color map for 'velocity'
FP_fdistribu = GetColorTransferFunction('fixed_point_fdistribu')
FP_fdistribu.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 29.205000000000002, 0.865003, 0.865003, 0.865003, 58.410000000000004, 0.705882, 0.0156863, 0.14902]
FP_fdistribu.ScalarRangeInitialized = 1.0

# show data from grid
gridDisplay = Show(producer, renderView1, 'StructuredGridRepresentation')

gridDisplay.Representation = 'Surface'
gridDisplay.ColorArrayName = ['POINTS', 'fixed_point_fdistribu']
gridDisplay.LookupTable = FP_fdistribu

# get color legend/bar for FP_fdistribu in view renderView1
FP_fdistribuColorBar = GetScalarBar(FP_fdistribu, renderView1)
FP_fdistribuColorBar.Title = 'fixed_point_fdistribu'
FP_fdistribuColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
FP_fdistribuColorBar.Visibility = 1

# show color legend
gridDisplay.SetScalarBarVisibility(renderView1, True)


# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# SetActiveView(renderView1)
# create extractor
# pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
# # trace defaults for the extractor.
# pNG1.Trigger = 'TimeStep'

# # init the 'PNG' selected for 'Writer'
# pNG1.Writer.FileName = 'screenshot_{timestep:06d}.png'
# pNG1.Writer.ImageResolution = [1600,800]
# pNG1.Writer.Format = 'PNG'

# ------------------------------------------------------------------------------
# Catalyst options
options = catalyst.Options()
## 0: no client, generate the images
## 1: live visualization
options.EnableCatalystLive = 1


# Greeting to ensure that ctest knows this script is being imported
def catalyst_execute(info):
    global producer
    producer.UpdatePipeline()
    print("-----------------------------------")
    print("executing (cycle={}, time={})".format(info.cycle, info.time))
    print("bounds:", producer.GetDataInformation().GetBounds())
    print("bidon-range:", producer.PointData["fixed_point_fdistribu"].GetRange(-1))
    # print("pressure-range:", producer.CellData["pressure"].GetRange(0))

    # In a real simulation sleep is not needed. We use it here to slow down the
    # "simulation" and make sure ParaView client can catch up with the produced
    # results instead of having all of them flashing at once.
    if options.EnableCatalystLive:
        time.sleep(1)
