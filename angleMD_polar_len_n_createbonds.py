import numpy as np
import pandas as pd

import os
import sys
from ovito.io import *
from ovito.modifiers import *

def transform_frame(p, frame):
    d = p.compute(frame)
    L = d.cell[0,0]
    step = d.attributes['Timestep']
    pos = np.array(d.particles['Position'])
    typ = np.array(d.particles['Particle Type'])
    mol = np.array(d.particles['Molecule Identifier'])
    ids = np.array(d.particles['Particle Identifier'])

    top = np.array(d.particles.bonds['Topology'])
    
    #print(top.shape)
    top_identifier = ids[top]
    #top = np.sort(top)
    #print(top_identifier[:3,:])
    top_sorted = np.vstack([(top*(1-np.argsort(top_identifier))).sum(axis = 1),(top*(np.argsort(top_identifier))).sum(axis = 1)]).T
    #print(top_identifier_sorted[:3,:])
    dvecs = np.empty((len(top), 2)) #for each bond store the bond vector
    dvecs = (pos[top_sorted[:,1]]-pos[top_sorted[:,0]])[:,:2]
    dvecs[dvecs > L/2] = dvecs[dvecs > L/2] - L #PBC...
    dvecs[dvecs <= -L/2] = dvecs[dvecs <= -L/2] + L
    drs = np.sqrt(dvecs[:,0]**2+dvecs[:,1]**2) #bond length
    urs = dvecs.copy() #store bond directions
    urs[:,0] /= drs 
    urs[:,1] /= drs
    angs = np.arctan2(urs[:,1],urs[:,0])/np.pi*180 #store angle of bond direction
    aps = np.empty(len(pos))
    dthetas = np.empty(len(pos))
    lens = np.empty(len(pos))
    Rs = np.empty(len(pos))
    for i in range(len(aps)):
        i0 = np.where(top_sorted[:,0] == i)[0]
        i1 = np.where(top_sorted[:,1] == i)[0]
        if len(i0) > 0 and len(i1) > 0:
            if angs[i0[0]] < -150 and angs[i1[0]] > 150 or angs[i0[0]] > 150 and angs[i1[0]] < -150:
                aps[i] = (angs[i0[0]]-angs[i1[0]])/2.
            else: aps[i]= np.nanmean([angs[i0[0]],angs[i1[0]]]) #the angle assigned to an atom is the mean bond angle
            dthetas[i] = angs[i0[0]] - angs[i1[0]]
            Rs[i] = 1./dthetas[i]
        else:
            dthetas[i] = np.nan
            Rs[i] = np.nan
            if len(i0) > 0:
                aps[i] = angs[i0[0]]
            elif len(i1) > 0:
                aps[i] = angs[i1[0]]

    for i in range(len(aps)):
        lens[i] = len(mol[mol==mol[i]])

    mols, mol_lens = np.unique(mol,return_counts= True)

    mol_dthetas = np.empty(len(mols))
    for i in range(len(mols)):
        if mol_lens[i] < 5: continue
        front_end = np.where((typ == 3)*(mol == i))[0]
        if len(front_end) == 0: continue
        back_end = np.where((typ == 2)*(mol == i))[0]
        front_end_bond = np.where(top_sorted[:,1]==front_end[0])[0]
        back_end_bond = np.where(top_sorted[:,0]==back_end[0])[0]
        mol_dthetas[i] = (angs[front_end_bond][0]-angs[back_end_bond][0])/(mol_lens[i]-2) #mmmm

    #here I was doing dtheta as mean over filaments
    """for i in range(len(typ)):
        if typ[i] != 3: continue
        for j in range(len(typ)):
            if typ[j] != 2: continue
            dthetas"""
    
    dthetas[dthetas<-180] += 360
    dthetas[dthetas>180] -= 360
    mol_dthetas[mol_dthetas<-180] += 360
    mol_dthetas[mol_dthetas>180] -= 360
    aps[aps < 0] += 360

    #pos = pos[typ < 6]
    #ids = ids[typ < 6]
    #mol = mol[typ < 6]
    #aps = aps[typ < 6]
    #typ = typ[typ < 6]

    return pos, ids, mol, aps, lens, typ, step, L, dthetas, mol_lens, mol_dthetas

path = sys.argv[1]

file_og = '%s/output.xyz'%(path) #attenzione!!
file_new = '%s/output_angles_len.xyz'%(path)
polar_file = '%s/polar.csv'%(path)

#bonds = '%s/bonds.dump'%(path)
p = import_file(file_og)
#mod = LoadTrajectoryModifier()
#p.modifiers.append(mod)
#mod.source.load(bonds, columns = ['Particle Identifiers.1', 'Particle Identifiers.2', 'Bond Type', 'Energy.1', 'Force.1', 'Length'], multiple_frames = True)
ff = p.source.num_frames

mod_bond = CreateBondsModifier(intra_molecule_only=True, cutoff=1.2)
p.modifiers.append(mod_bond)

f = open(file_new, 'w')
fp = open(polar_file, 'w')
fp.write('step,norm,angle,density,length,dtheta\n')
out = open('analysis_out.txt', 'w')
analyse_every = 1
for i in range(1,ff, analyse_every):
    pos, ids, mol, a, lens, typ, step, L, dthetas, mol_lens, mol_dthetas = transform_frame(p, i)
    f.write("ITEM: TIMESTEP\n")
    f.write("%d\n"%(step))
    f.write("ITEM: NUMBER OF ATOMS\n")
    f.write("%d\n"%(len(ids)))
    f.write("ITEM: BOX BOUNDS pp pp pp\n")
    f.write("-%.1f %.1f\n"%(L/2,L/2))
    f.write("-%.1f %.1f\n"%(L/2,L/2))
    f.write("-%.1f %.1f\n"%(L/2,L/2))
    f.write("ITEM: ATOMS id type mol x y z a len dtheta\n")
    for j in range(len(ids)):
        f.write("%d %d %d %.2f %.2f %.2f %.2f %d %.2f\n"%(ids[j],typ[j],mol[j],pos[j,0],pos[j,1],pos[j,2],a[j], lens[j], dthetas[j]))
    if len(ids) > 0:
        means = np.array([np.mean(np.sin(a[:]/360*2*np.pi)),  np.mean(np.cos(a[:]/360*2*np.pi))])
        density = len(ids)/L/L
        mean_length = np.mean(mol_lens[:])
        mean_dtheta = np.nanmean(dthetas)
        mean_mol_dtheta = np.nanmean(mol_dthetas)
        mean_R = np.nanmean(1./dthetas)
        mean_mol_R = np.nanmean(1./mol_dthetas)
        fp.write('%d,%f,%f,%f,%f,%f\n'%(step, np.linalg.norm(means), np.arctan2(means[0], means[1]), density, mean_length, mean_dtheta))
    print('done ',i)
    
out.close()
f.close()
fp.close()
