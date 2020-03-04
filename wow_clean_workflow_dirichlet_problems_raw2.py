cellSize = 0.05
radius = 1.
from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm, Viewer
from fipy.tools import numerix
mesh = Gmsh2D('''
cellSize = %(cellSize)g;
radius = %(radius)g;
Point(1) = {0,0,0, cellSize};
Point(2) = {-radius,0,0,cellSize};
Point(3) = {0,radius,0,cellSize};
Point(4) = {radius,0,0,cellSize};
Point(5) = {0,-radius,0,cellSize};
Circle(6) = {2,1,3};
Circle(7) = {3,1,4};
Circle(8) = {4,1,5};
Circle(9) = {5,1,2};
Line Loop(10) = {6,7,8,9};
''' % locals())
mesh
mesh = Gmsh2D('''
cellSize = %(cellSize)g;
radius = %(radius)g;
Point(1) = {0,0,0, cellSize};
Point(2) = {-radius,0,0,cellSize};
Point(3) = {0,radius,0,cellSize};
Point(4) = {radius,0,0,cellSize};
Point(5) = {0,-radius,0,cellSize};
Circle(6) = {2,1,3};
Circle(7) = {3,1,4};
Circle(8) = {4,1,5};
Circle(9) = {5,1,2};
Line Loop(10) = {6,7,8,9};
''' % locals())
ls
pwd
mesh = Gmsh2D('''
cellSize = %(cellSize)g;
radius = %(radius)g;
Point(1) = {0,0,0, cellSize};
Point(2) = {-radius,0,0,cellSize};
Point(3) = {0,radius,0,cellSize};
Point(4) = {radius,0,0,cellSize};
Point(5) = {0,-radius,0,cellSize};
Circle(6) = {2,1,3};
Circle(7) = {3,1,4};
Circle(8) = {4,1,5};
Circle(9) = {5,1,2};
Line Loop(10) = {6,7,8,9};
Plane Surface(11) = {10};
''' % locals())
mesh
phi = CellVariable(name = "solution variable", mesh=mesh, value = 0.)
from fipy import input
viewer = Viewer(vars=phi, datamin=-1, datamax=1.)
viewer.plotMesh()
pyplot.ion()
from matplotlib import pyplot
pyplot.ion()
viewer
viewer.plotMesh()
viewer=None; viewer = Viewer(vars=phi, datamin=-1.,datamax=1.); viewer.plotMesh()
fig = pyplot.gcf()
ax = fig.gca()
ax.artists
ax.artists[0].remove()
ax.artists[1].remove()
ax.artists[0].remove()
ax.artists
ax.containers
ax.contours
ax.lines
phi
viewer.plotMesh?
viewer.plotMesh()
Viewer?
Viewer?
viewer=None; viewer = Viewer(vars=phi, datamin=-1.,datamax=1.); viewer.plotMesh()
ax
ax = pyplot.gca()
ax
ax.mesh
ax.collections
ax.collections[0].remove()
ax.containers
ax.patches
ax.lines
ax.stackplot?
ax.child_axes
mesh
import fipy
viewer = fipy.Matplotlib2DViewer(vars=phi)
viewer.plotMesh()
pyplot.ion()
viewer = fipy.Matplotlib2DViewer(vars=phi, datamin=-1, datamax=1)
mesh
mesh.VTKCellDataSet
mesh.cellCenters
mesh.cellCenters.shape
ax = pyplot.gca()
mesh.vertexCoords
vcoord = mesh.vertexCoords
vcoord.shape
ax.scatter(vcoord[0], vcoord[1], c='k', s=2, marker='.')
fig = pyplot.gcf()
fig.axes
ax = fig.axes[1]
ax.scatter(vcoord[0], vcoord[1], c='k', s=2, marker='.')
ax.lines
ax.collections
ax.collections[-1].remove()
ax.collections[-1].remove()
cbar = fig.axes[1]
ax = fig.axes[0]
ax.scatter(vcoord[0], vcoord[1], c='k', s=2, marker='.')
mesh.faceVertexIDs
thingy = mesh.faceVertexIDs
thingy.shape
mesh.physicalFaces
mesh.VTKFaceDataSet
mesh.VTKCellDataSet
mesh.vertexCoords
mesh.topology
mesh.physicalCells
mesh.numberOfFaces
mesh.faceVertexIDs
mesh.faceVertexIDs.max()
mesh.faceVertexIDs.min()
vc.shape
vcoord.shape
mesh.exteriorFaces
po = np.where( mesh.exteriorFaces )
import numpy as np
po = np.where( mesh.exteriorFaces.value )
po
mesh.faceCellIDs
mesh.faceCellIDs.shape
mesh.cellFaceIDs
mesh.cellFaceIDs.shape
mesh.facesDown
mesh.faceVertexIDs
exterior_faces = mesh.exteriorFaces
exterior_faces
exterior_faces = mesh.exteriorFaces.value
exterior_faces
lala = mesh.faceVertexIDs[:,exterior_faces]
lala
lala.T
vcoord.shape
for row in lala.T:
    ax.plot(vcoord[0,row], vcoord[1,row], c=[0.6,0.6,0.6], lw=0.5)
for row in lala.T:
    ax.plot(vcoord[0,row], vcoord[1,row], c=[1,0.6,0.6], lw=0.5)
for row in lala.T:
    ax.plot(vcoord[0,row], vcoord[1,row], c=[1,0.6,0.6], lw=3)
mesh.cellFaceIDs
ax.lines
for l in ax.lines[::-1]:
    l.remove()
mesh.cellFaceIDs
mesh.cellFaceIDs[mesh.faceVertexIDs]
mesh.cellFaceIDs[:,mesh.faceVertexIDs]
mesh.cellFaceIDs[:,mesh.faceVertexIDs].shape
mesh.cellFaceIDs[:,mesh.faceVertexIDs][0]
mesh.cellFaceIDs[:,mesh.faceVertexIDs].T
mesh.cellFaceIDs[:,mesh.faceVertexIDs].T[0]
mesh.cellFaceIDs[:,mesh.faceVertexIDs[0]]
mesh.cellFaceIDs[mesh.faceVertexIDs[0]]
mesh.cellFaceIDs[:,mesh.faceVertexIDs[0]]
mesh.cellFaceIDs[:,mesh.faceVertexIDs[0]].shape
mesh.cellFaceIDs.shape
mesh.faceVertexIDs[0]
mesh.faceVertexIDs[0].shape
mesh.faceVertexIDs.shape
mesh.cellFaceIDs[:,mesh.faceVertexIDs[:,0]].shape
mesh.cellFaceIDs[:,mesh.faceVertexIDs[:,0]]
vcoord[ :, mesh.cellFaceIDs[:,mesh.faceVertexIDs[:,0]]]
vcoord[ mesh.cellFaceIDs[:,mesh.faceVertexIDs[:,0]]]
vcoord[ mesh.cellFaceIDs[:,mesh.faceVertexIDs[:,0]]]
mesh.cellFaceIDs[:,mesh.faceVertexIDs[:,0]]]
mesh.cellFaceIDs[:,mesh.faceVertexIDs[:,0]]
for row in mesh.cellFaceIDs[:,mesh.faceVertexIDs[:,0]]:
    ax.plot(vcoord[0,row], vcoord[1,row], c='r', lw=2)
for row in mesh.cellFaceIDs[:,mesh.faceVertexIDs[:,0]]:
    ax.plot(vcoord[row,0], vcoord[row,1], c='r', lw=2)
for row in mesh.cellFaceIDs[:,mesh.faceVertexIDs[:,0]].T:
    ax.plot(vcoord[row,0], vcoord[row,1], c='r', lw=2)
for row in mesh.cellFaceIDs[:,mesh.faceVertexIDs[:,0]].T:
    ax.plot(vcoord[0,row], vcoord[1,row], c='r', lw=2)
row
for l in ax.lines[::-1]:
    l.remove()
mesh.cellFaceIDs[:,mesh.faceVertexIDs[:,0]]
mesh.cellFaceIDs[:,0]
mesh.faceVertexIDs[mesh.cellFaceIDs[:,0]]
mesh.faceVertexIDs[:,mesh.cellFaceIDs[:,0]]
moo = mesh.faceVertexIDs[:,mesh.cellFaceIDs[:,0]]
vcoord[moo[0]]
vcoord[0,moo[0]]
vcoord[0,moo[1]]
ax.plot(vcoord[0,moo[0]], vcoord[1,moo[0]], c='r', lw=2)
for thing in mesh.cellFaceIDs.T:
    moo = mesh.faceVertexIDs[:,thing]
    ax.plot(vcoord[0,moo[0]], vcoord[1,moo[0]], c=[0.5,0,0], lw=2)
for thing in mesh.cellFaceIDs.T:
    moo = mesh.faceVertexIDs[:,thing]
    ax.plot(vcoord[0,moo[1]], vcoord[1,moo[1]], c=[0.5,0,0], lw=2)
[l.remove() for l in ax.lines[::-1]]
mesh.cellFaceIDs[0]
mesh.cellFaceIDs[:,0]
mesh.faceVertexIDs.shape
mesh.faceVertexIDs[:,mesh.cellFaceIDs[:,0]]
mesh.faceVertexIDs[:,list(mesh.cellFaceIDs[:,0])]
for thing in mesh.cellFaceIDs.T:
    moo = mesh.faceVertexIDs[:,thing]
    for row in moo.T:
        ax.plot(vcoord[0,row], vcoord[1,row], c=[0.3,0.3,0.3], lw=1)
%history -f wow_some_progress_raw.py
D = 1.
eq = TransientTerm() = DiffusionTerm(coeff=D)
eq = TransientTerm() == DiffusionTerm(coeff=D)
X,Y = mesh.faceCenters
X
Y
X.shape
Y.shape
vcoords.shape
vcoord.shape
ax.scatter(X,Y, c='w', s=4, marker='.')
ax.scatter(X,Y, c='w', s=4, marker='.', zorder=1000)
phi.constrain?
phi.constrain(X, mesh.exteriorFaces)
dt = 10*0.9*cellSize**2/(2*D)
steps = 10
for step in range(steps):
    eq.solve(var=phi, dt=dt)
    viewer.plot()
    pyplot.pause(1)
DiffusionTerm(coeff=D).solve(var=phi)
viewer.plot()
ax
ax.containers
ax.mesh
ax.meshes
ax.artists
ax.patches
ax.collections
ax.collections[0]
mo = ax.collections[0]
mo.colors
mo.set_visible(False)
mo.set_visible(True)
mo.properties()
mo.colors
mo.colorbar
mo.cmap
mo.cmap = pyplot.cm.Greys
mo.set_cmap(pyplot.cm.Greys)
mo.update()
pyplot.ion()
mo.get_cmap
mo.get_cmap()
thing = mo.get_cmap()
thing(0)
thing(255)
thing(30)
thing(200)
mo.check_update()
mo.update_scalarmappable()
mo.set_facecolors?
mo.set_facecolors('r')
phi.value
phi.value.shape
phi.min()
phi.max()
mo.set_facecolors(pyplot.cm.inferno((1+phi)/2.))
ax.lines
for l in ax.lines[::-1]:
    l.remove()
ax.scatter([],[])
ax.collections
for j in range(4):
    ax.collections[-1].remove()
ax.lines
ax.artists
ax.containers
%history -f more_progress_raw.py
########################
# wowee
# start over; try to do a triangle.
tmesh = Gmsh2D('''
cellSize = %(cellSize)g;
Point(1) = {-1,0,0,cellSize};
Point(2) = {1,0,0,cellSize};
Point(3) = {0,1.732,0,cellSize};
Edge(4) = {1,1,2};
Edge(5) = {2,1,3};
Edge(6) = {3,1,1};
Line Loop(7) = {4,5,6};
Plane Surface(8) = {7};
''' % (cellSize,))
tmesh = Gmsh2D('''
cellSize = %(cellSize)g;
Point(1) = {-1,0,0,cellSize};
Point(2) = {1,0,0,cellSize};
Point(3) = {0,1.732,0,cellSize};
Edge(4) = {1,1,2};
Edge(5) = {2,1,3};
Edge(6) = {3,1,1};
Line Loop(7) = {4,5,6};
Plane Surface(8) = {7};
''' %cellSize)
tmesh = Gmsh2D('''
cellSize = 0.05;
Point(1) = {-1,0,0,cellSize};
Point(2) = {1,0,0,cellSize};
Point(3) = {0,1.732,0,cellSize};
Edge(4) = {1,1,2};
Edge(5) = {2,1,3};
Edge(6) = {3,1,1};
Line Loop(7) = {4,5,6};
Plane Surface(8) = {7};
''')
tmesh = Gmsh2D('''
cellSize = 0.05;
Point(1) = {-1,0,0,cellSize};
Point(2) = {1,0,0,cellSize};
Point(3) = {0,1.732,0,cellSize};
Edge(4) = {1,1,2};
Edge(5) = {2,1,3};
Edge(6) = {3,1,1};
Line Loop(7) = {4,5,6};
Plane Surface(8) = {7};
''')
locals()
cellSize, tHeight = 0.05, np.sqrt(3)
tmesh = Gmsh2D('''
cellSize = %(cellSize)g;
tHeight = %(tHeight)g;
Point(1) = {-1,0,0,cellSize};
Point(2) = {1,0,0,cellSize};
Point(3) = {0,tHeight,0,cellSize};
Edge(4) = {1,1,2};
Edge(5) = {2,1,3};
Edge(6) = {3,1,1};
Line Loop(7) = {4,5,6};
Plane Surface(8) = {7};
''' % locals())
cellSize, tHeight = 0.05, np.sqrt(3)
tmesh = Gmsh2D('''
cellSize = %(cellSize)g;
tHeight = %(tHeight)g;
Point(1) = {-1,0,0,cellSize};
Point(2) = {1,0,0,cellSize};
Point(3) = {0,tHeight,0,cellSize};
Line(4) = {1,1,2};
Line(5) = {2,1,3};
Line(6) = {3,1,1};
Line Loop(7) = {4,5,6};
Plane Surface(8) = {7};
''' % locals())
cellSize, tHeight = 0.05, np.sqrt(3)
tmesh = Gmsh2D('''
cellSize = %(cellSize)g;
tHeight = %(tHeight)g;
Point(1) = {-1,0,0,cellSize};
Point(2) = {1,0,0,cellSize};
Point(3) = {0,tHeight,0,cellSize};
Line(4) = {1,1,2};
Line(5) = {2,1,3};
Line(6) = {1,1,3};
Line Loop(7) = {4,5,6};
Plane Surface(8) = {7};
''' % locals())
cellSize, tHeight = 0.02, np.sqrt(3)
tmesh = Gmsh2D('''
cellSize = %(cellSize)g;
tHeight = %(tHeight)g;
Point(1) = {-1,0,0,cellSize};
Point(2) = {1,0,0,cellSize};
Point(3) = {0,tHeight,0,cellSize};
Line(4) = {1,1,2};
Line(5) = {2,1,3};
Line(6) = {1,1,3};
Line Loop(7) = {4,5,6};
Plane Surface(8) = {7};
''' % locals())
cellSize, tHeight = 0.02, np.sqrt(3)
tmesh = Gmsh2D('''
cellSize = %(cellSize)g;
tHeight = %(tHeight)g;
Point(1) = {-1,0,0,cellSize};
Point(2) = {1,0,0,cellSize};
Point(3) = {0,tHeight,0,cellSize};
Line(4) = {1,1,2};
Line(5) = {2,1,3};
Line(6) = {1,1,3};
Line Polygon(7) = {4,5,6};
Plane Surface(8) = {7};
''' % locals())
cellSize, tHeight = 0.02, np.sqrt(3)
tmesh = Gmsh2D('''
cellSize = %(cellSize)g;
tHeight = %(tHeight)g;
Point(1) = {-1,0,0,cellSize};
Point(2) = {1,0,0,cellSize};
Point(3) = {0,tHeight,0,cellSize};
Line(4) = {1,2};
Line(5) = {2,3};
Line(6) = {1,3};
Line Polygon(7) = {4,5,6};
Plane Surface(8) = {7};
''' % locals())
cellSize, tHeight = 0.02, np.sqrt(3)
tmesh = Gmsh2D('''
cellSize = %(cellSize)g;
tHeight = %(tHeight)g;
Point(1) = {-1,0,0,cellSize};
Point(2) = {1,0,0,cellSize};
Point(3) = {0,tHeight,0,cellSize};
Line(4) = {1,2};
Line(5) = {2,3};
Line(6) = {3,1};
Line Polygon(7) = {4,5,6};
Plane Surface(8) = {7};
''' % locals())
tmesh
tphi = CellVariable(name = "solution variable", mesh=mesh, value=0.)
DiffusionTerm?
forcing = CellVariable(mesh=mesh, name='pressure')
forcing = CellVariable(mesh=mesh, name='pressure', value=-2)
eq = DiffusionTerm(coeff=1.) + forcing == 0
tphi.constrain(0., mesh.exteriorFaces)
eq.solve(var=phi)
tviewer = Viewer(vars=phi)
tviewer = Viewer(vars=tphi)
eq.solve(var=tphi)
tviewer = Viewer(vars=tphi)
forcing = CellVariable(mesh=tmesh, name='pressure', value=-2)
eq = DiffusionTerm(coeff=1.) + forcing == 0
eq.solve(var=tphi)
tphi.constrain(0., tmesh.exteriorFaces)
tphi = CellVariable(name = "solution variable", mesh=tmesh, value=0.)
tphi.constrain(0., tmesh.exteriorFaces)
eq.solve(var=tphi)
cellSize, tHeight = 0.02, np.sqrt(3)
tmesh = Gmsh2D('''
cellSize = %(cellSize)g;
tHeight = %(tHeight)g;
Point(1) = {-1,0,0,cellSize};
Point(2) = {1,0,0,cellSize};
Point(3) = {0,tHeight,0,cellSize};
Line(4) = {1,2};
Line(5) = {2,3};
Line(6) = {3,1};
Line Polygon(7) = {4,5,6};
Plane Surface(8) = {7};
''' % locals())
tphi = CellVariable(name = "solution variable", mesh=tmesh, value=0.)
viewer = Viewer(vars=tphi, datamin=-1, datamax=1)
tphi.constrain(0., tmesh.exteriorFaces)
forcing = CellVariable(mesh=tmesh, name='pressure', value=-2)
eq = DiffusionTerm(coeff=1.) - forcing == 0
eq.solve(var=tphi)
viewer.plot()
viewer = Viewer(vars=phi, datamin=0., datamax=phi.max())
tphi.max()
tphi.values
tphi.value
viewer = Viewer(vars=tphi, datamin=0., datamax=tphi.value.max())
fig = pyplot.gcf()
fig.axes
ax = fig.axes[0]
ax.collections
ax.collections[0].set_facecolors(pyplot.cm.inferno(tphi.value))
ax.lines
ax.collections
ax.artists
ax.child_axes
ax.containers
ax.sticky_edges
ax.texts
ax.tables
ax.markers
ax.patches
fig.patches
fig.lines
fig.artists
fig.collections
mo.set_edgecolor([1,1,1,0.3])
mo.set_edgecolors('w')
mo.get_facecolors()
mo.get_edgecolors()
pyplot.ion()
tax
ax = pyplot.gca()
ax
fig
ax = fig.axes[0]
ax
ax.collections
ax.collections[0].set_edgecolors([1,1,1,0.2])
ax.collections[0].set_edgecolors([1,1,1,0.])
tphi.max()
tphi.value
tphi.value.max()
ax.collections[0].set_facecolors(pyplot.cm.inferno(9./2*tphi.value))
%history -f some_more_progress_raw.py
import cmocean
ax.collections[0].set_facecolors(cmocean.cm.ice(9./2*tphi.value))
ax.collections[0].set_edgecolors('g')
ax.collections[0].set_edgecolors([0,1,0,0.2])
cellSize, tHeight = 0.1, np.sqrt(3)
tmesh = Gmsh2D('''
cellSize = %(cellSize)g;
tHeight = %(tHeight)g;
Point(1) = {-1,0,0,cellSize};
Point(2) = {1,0,0,cellSize};
Point(3) = {0,tHeight,0,cellSize};
Line(4) = {1,2};
Line(5) = {2,3};
Line(6) = {3,1};
Line Polygon(7) = {4,5,6};
Plane Surface(8) = {7};
''' % locals())
tphi = CellVariable(name = "solution variable", mesh=tmesh, value=0.)
tphi.constrain(0., tmesh.exteriorFaces)
tphi.constrain(0., tmesh.exteriorFaces) # dirichlet boundary condition
forcing = CellVariable(mesh=tmesh, name='pressure', value=-2)
eq = DiffusionTerm(coeff=1.) == forcing # set up poisson eqn
eq.solve(var=tphi)
viewer = Viewer(vars=tphi, datamin=0., datamax=tphi.value.max()) # vis
ax.collections[0].set_facecolors(cmocean.cm.ice(9./2*tphi.value)) # for fun
fig = pyplot.gcf()
ax = fig.axes[0]
ax.collections[0].set_facecolors(cmocean.cm.ice(9./2*tphi.value)) # for fun
ax.collections[0].set_edgecolors([0.8,0.8,0.8,0.1])
%history -f wow_clean_workflow_dirichlet_problems_raw.py
fig.axes
fig.axes[1].remove()
fig.tight_layout()
ax.axis('square')
fig.tight_layout()
fig.set_figheight(8)
fig.set_figwidth(8)
fig.tight_layout()
fig.savefig('triangle_poiseuille_fipy.png', dpi=120, bbox_inches='tight')
%history -f wow_clean_workflow_dirichlet_problems_raw2.py
