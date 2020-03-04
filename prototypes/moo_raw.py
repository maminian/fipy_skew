run fipy_triangle_solve.py
tphi.faceConstraints
tmesh.exteriorFaces
tmesh.faceNormals
fipy.boundaryConditions.FixedFlux?
theta = CellVariable(name = "cell problem", mesh=tmesh, value=0.)
celleq = (DiffusionTerm(coeff=1.) == 0.)
fipy.boundaryConditions.FixedFlux(tmesh.faceNormals.data, tphi.value[tmesh.exteriorFaces])
fipy.boundaryConditions.FixedFlux(tmesh.faceNormals.data, tphi.value[tmesh.exteriorFaces])
tmesh.faceNormals.data
fipy.boundaryConditions.FixedFlux?
fipy.boundaryConditions.FixedFlux(np.where(tmesh.faceNormals.mask)[0], tphi.value[tmesh.exteriorFaces])
fipy.boundaryConditions.FixedFlux(tmesh.exteriorFaces, tphi.value[tmesh.exteriorFaces])
fipy.boundaryConditions.FixedFlux(np.where(tmesh.exteriorFaces)[0], tphi.value[tmesh.exteriorFaces])
tmesh.exteriorFaces.shape
tmesh.exteriorFaces
%history -f moo_raw.py
