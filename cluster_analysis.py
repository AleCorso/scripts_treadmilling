import numpy as np
import pandas as pd
import os
import sys
import scipy.ndimage
from scipy.stats import binned_statistic, binned_statistic_2d
from scipy.sparse import csgraph
import scipy.sparse
from scipy.spatial import cKDTree
from skimage.morphology import medial_axis
#import matplotlib.pyplot as plt
import skan
from ovito.io import *
from ovito.modifiers import *

#input arguments: path to xyz file
path = sys.argv[1] 
os.system('mkdir '+path+'/cycles')
file_post = '%s/output_angles_len.xyz'%(path)
file_out = '%s/cycles.csv'%(path)
outfile = open('%s/output_clustered.xyz'%(path),'w+')
p = import_file(file_post)

# angles are rescaled between 0 and 2*np.pi*rescale_factor to balance distance and angle mismatch in clustering
rescale_factor = 5

# binning size of binned image used to find the skeleton
binsize = 2

def periodic_difference(v, b):
    s = np.sign(v)
    return v - (s * v > b / 2) * s * b

def cluster_pos_angle(pos,delta_pos,boxsize,boxorigin):
    tree = cKDTree((pos-boxorigin) % boxsize, boxsize=boxsize)
    sm = tree.sparse_distance_matrix(tree, max_distance=delta_pos, output_type="coo_matrix")
    lb_n, lb = scipy.sparse.csgraph.connected_components(sm, return_labels=True)
    return lb_n,lb

def dfs(graph,node, parent, visited, cycle,debug=False):
    if debug:print("dfs: "+str(node))
    visited[node] = True
    if debug: print(graph[node,:].indices)
    for neig in graph[node,:].indices:
        if debug:print(neig)
        w = graph[node,neig]
        if not visited[neig]:
            parent[neig] = node
            cycle_length, rest_of_cycle, stop = dfs(graph, neig, parent, visited, cycle,debug)
            if cycle_length > 0:
                if neig == stop: 
                    if debug:print("returning from useless node "+str(node)+" with length "+str(cycle_length)+" and path: "+str(rest_of_cycle))
                    return cycle_length, rest_of_cycle, node
                else:
                    if debug:print("returning from node "+str(node)+" with length "+str(cycle_length+w)+" and path: "+str(rest_of_cycle + [neig]))
                    return cycle_length + w, rest_of_cycle + [neig], stop
        elif neig != parent[node]:
            if debug:print("found cycle!")
            return w, [neig], neig
    if debug:print("nope")
    return 0,[], None
def max_cycle_length(graph,debug = False):
    max_cycle = -1
    N = graph.shape[0]
    for i in range(1):
        if debug:print("starting with node: "+str(i))
        visited = [False]* N
        parent = [-1] * N
        cycle_length,cycle_temp, stop = dfs(graph, i, parent, visited,[],debug)
        if cycle_length> max_cycle:
            max_cycle = cycle_length
            cycle = cycle_temp
    return max_cycle, cycle

mod_orient = ComputePropertyModifier(
    output_property = 'Position',
    expressions= ['Position.X', 'Position.Y','a/360*6.283*'+str(rescale_factor)]
)

mod_cluster = ClusterAnalysisModifier(
    cutoff=2.0, 
    sort_by_size=True,
    compute_com = True,
    compute_gyration = True,
    only_selected = True)

mod = LoadTrajectoryModifier()
mod.source.load(file_post)

def autocorr_naive_nan(x):
    N = len(x)
    e_x = np.array([np.sin(2*np.pi*x/360),np.cos(2*np.pi*x/360)])
    return [np.array([np.nanmean((e_x[:,iSh:] * e_x[:,:N-iSh]).sum(axis=0)) for iSh in range(N)]), np.array([np.sum(~np.isnan((e_x[:,iSh:] * e_x[:,:N-iSh]).sum(axis=0))) for iSh in range(N)])]

def lowerquart(a):
    return np.nanpercentile(a,25)
def upperquart(a):
    return np.nanpercentile(a,75)

p.modifiers.append(mod)

p.modifiers.append(ComputePropertyModifier(expressions=['1'], output_property='Unity'))
p.modifiers.append(ExpressionSelectionModifier(expression ='ParticleType < 4'))

p.modifiers.append(mod_orient) #modifies how Position is computed, to have as z dimension the cos of a between rescaled
p.modifiers.append(mod_cluster) #clusters based on position

widths = np.zeros((p.source.num_frames, 40))
len_dist = np.zeros((p.source.num_frames, 40,3))
cycles = np.zeros((p.source.num_frames))

#persistence = np.zeros((p.source.num_frames, int(Lx),2))

df = pd.DataFrame({'step':[],'cluster_id':[],'num_particles':[],'cycle_length':[], 'mean_width':[],'mean_fil_len':[],'mean_dtheta':[],'x_pos':[],'y_pos':[]})

for f in range(1,p.source.num_frames,1): #analyse one every 1 frames now
    print('frame '+str(f)+' of '+str(p.source.num_frames))

    d = p.compute(f)
    d.cell_[:,2] = (0, 0, 2.*np.pi*rescale_factor)
    d.cell_[2,3] = np.pi*rescale_factor
    Lx = d.cell[0,0]
    Ly = d.cell[1,1]
    nclust, labels = cluster_pos_angle(d.particles.positions[d.particles['Particle Type']<4],2,np.array([Lx,Ly,2*np.pi*rescale_factor]),boxorigin = d.cell[:,3])
    #print(nclust)

    pos = np.array(d.particles['Position'])
    typ = np.array(d.particles['Particle Type'])
    mol = np.array(d.particles['Molecule Identifier'])
    a = np.array(d.particles['a'])
    ids = d.particles['Particle Identifier']
    dthetas = d.particles['dtheta']
    lens = d.particles['len']
    all_labels = []
    conta = 0
    for i in range(len(ids)):
        if d.particles['Particle Type'][i] < 4:
            all_labels.append(labels[conta])
            conta += 1
        else: all_labels.append(-1)

    outfile.write("ITEM: TIMESTEP\n")
    outfile.write("%d\n"%(d.attributes['Timestep']))
    outfile.write("ITEM: NUMBER OF ATOMS\n")
    outfile.write("%d\n"%(len(ids)))
    outfile.write("ITEM: BOX BOUNDS pp pp pp\n")
    outfile.write("-%.1f %.1f\n"%(Lx/2,Lx/2))
    outfile.write("-%.1f %.1f\n"%(Ly/2,Ly/2))
    outfile.write("-%.1f %.1f\n"%(3.14159*rescale_factor,3.14159*rescale_factor))
    outfile.write("ITEM: ATOMS id type mol x y z a len dtheta cluster\n")
    for j in range(len(ids)):
        outfile.write("%d %d %d %.2f %.2f %.2f %.2f %d %.2f %d\n"%(ids[j],typ[j],mol[j],pos[j,0],pos[j,1],pos[j,2],a[j], lens[j], dthetas[j], all_labels[j]))
    
    lens_time = np.array([0])
    distlen_time =np.array([0])
    dens_time =np.array([0]) 
    dists_time =  np.array([0])
    dthetas_time =  np.array([0])

    for i in range(nclust):
        data = d.particles.positions[d.particles.cluster == i+1]
        lens = d.particles['len'][d.particles.cluster == i+1]
        angles = d.particles['a'][d.particles.cluster == i+1]
        dthetas = d.particles['dtheta'][d.particles.cluster == i+1]

        if len(data) < 300 : break
        print('\t cluster '+str(i)+' of '+str(nclust)+': '+str(len(data))+' particles')

        #PBC CHECK
        #if cluster splitted by periodic boundary, this will translate one part to the other

        if np.logical_and(np.min(data[:,0])<-Lx/2+4,np.max(data[:,0])>Lx/2-4):
            #print('pbc_x', f, ' ',i)
            binx = np.arange(-Lx/2, Lx/2,10)
            hist = np.histogram(data[:,0], bins=binx)
            if np.min(hist[0]) <1:
                void_x = np.argmin(hist[0][:]>0)
                translate_mask = data[:,0]>hist[1][void_x]
                #print('translating_x ',translate_mask.sum())
                #print('total ',data.shape[0])
                data[translate_mask,0]-=Lx
                #print('max_x ', np.max(data[:,0]))

        if np.logical_and(np.min(data[:,1]) < -Ly/2 + 4 , np.max(data[:,1]) > Ly/2 - 4):
            #print('pbc_y', f, ' ',i)
            binx = np.arange(-Ly/2, Ly/2,10)
            hist = np.histogram(data[:,1], bins=binx)
            if np.min(hist[0]) <1:
                void_y = np.argmin(hist[0][:]>0)
                translate_mask = data[:,1]>hist[1][void_y]
                #print('translating_y', translate_mask.sum())
                data[translate_mask,1]-=Ly
                #print('max_y ', np.max(data[:,1]))

        #BINARIZATION
        #transform coordinates to binary image
        binx = np.arange(np.min(data[:,0])-2, np.max(data[:,0])+2, binsize)
        biny = np.arange(np.min(data[:,1])-2, np.max(data[:,1])+2, binsize)
        binx_l = np.arange(np.min(data[:,0])-2, np.max(data[:,0])+2, binsize)
        biny_l = np.arange(np.min(data[:,1])-2, np.max(data[:,1])+2, binsize)
        x_binned = np.digitize(data[:,0],binx)
        y_binned = np.digitize(data[:,1],biny)
        data_binary = np.histogram2d(data[:,0], data[:,1],bins=[binx, biny])[0]
        data_filled = scipy.ndimage.binary_closing(data_binary, iterations=2)

        #DISTANCE TRANSFORM
        #computes the skeleton
        skel, distance = medial_axis(data_filled, return_distance=True)
        dist_on_skel = distance * skel
        dists = np.histogram(binsize*dist_on_skel.flatten(), bins = np.arange(41)+0.5)[0]
        widths[f,:]+= dists
        

        data_len = binned_statistic_2d(data[:,0], data[:,1], lens, 'mean', bins=[binx, biny])
        data_a = binned_statistic_2d(data[:,0], data[:,1], angles, 'mean', bins=[binx, biny])
        data_dtheta = binned_statistic_2d(data[:,0], data[:,1], dthetas, 'mean', bins=[binx_l, biny_l])

        #DISTANCE-LENGTH CORRELATION        
        mask = data_len.statistic.flatten()>2.
        distlen_time= np.hstack([distlen_time,(binsize*distance.flatten()[mask])])
        lens_time= np.hstack([lens_time,data_len.statistic.flatten()[mask]])
        dthetas_time = np.hstack([dthetas_time,data_dtheta.statistic.flatten()[mask]])
        #lens_time.append(data_len.statistic.flatten()[mask][:])
        #print(lens_time.shape)
        #print(distlen_time.shape)

        #mask = data_binary.flatten() > 0.8
        
        dens_time= np.hstack([dens_time,data_binary.flatten()[mask]])
        dists_time= np.hstack([dists_time,binsize*distance.flatten()[mask]])

        #SKeleton ANalysis
        #loads the skeleton, gets the coordinates of the longest path
        #print(skel)
        try: skeleton = skan.csr.Skeleton(skel)#, source_image=data_filled)
        except ValueError: continue
        branch_data= skan.summarize(skan.Skeleton(skel), find_main_branch=True)

        main_branches = branch_data.index[branch_data['main'] == True].tolist()
        path_x = np.array([0])
        path_y = np.array([0])
        for b in main_branches:
            path_x = np.hstack([path_x,skeleton.path_coordinates(b)[:,0]])
            path_y = np.hstack([path_y,skeleton.path_coordinates(b)[:,1]])

        #computes angle along the longest path
        a_spine = data_a.statistic[path_x[1:], path_y[1:]]

        #autocorrelation of the angle 
        """if np.nanmax(np.absolute(np.diff(a_spine)*6.28/360*0.8))<1:
            acorr, acorr_counts=autocorr_naive_nan(a_spine)
            if len(acorr)> Lx:
                acorr = acorr[:int(Lx)]
                acorr_counts = acorr_counts[:int(Lx)]
            persistence[f,:len(acorr),0]+=acorr*acorr_counts
            persistence[f,:len(acorr),1]+=acorr_counts"""

        if i==0: print('\t largest cluster max halfwidth ', np.max(binsize*dist_on_skel.flatten()))

        allpaths_x = np.array([0])
        allpaths_y = np.array([0])
        for b in branch_data.index.tolist():
            allpaths_x = np.hstack([allpaths_x,skeleton.path_coordinates(b)[:,0]])
            allpaths_y = np.hstack([allpaths_y,skeleton.path_coordinates(b)[:,1]])

        if (np.histogram(data_a.statistic[allpaths_x[1:], allpaths_y[1:]], bins = 30, range = (0,360))[0] >= 1).all():
            # This if selects all clusters that span all angles as candidate cycles. Note that also unclosed very curved clusters may be selected
            print('found a candidate cycle!')

            # Here i look for a cycle in the graph that spans all angles (the true cycle): iteratively i find a cycle using max, check if it is the true cycle, 
            # otherwise erase one node in it. Thanks to how the graph looks like in our specific case, this allows to eventually find the true cycle
            graph, (x_coords, y_coords) = skan.csr.skeleton_to_csgraph(skel, spacing = 1)
            max_cycle, cycle= max_cycle_length(graph)
            while (not (np.histogram(data_a.statistic[x_coords[cycle[:]], y_coords[cycle[:]]], bins = 10, range = (0,360))[0] >= 1).all()) and len(cycle)>2:
                #sns.scatterplot(x =binx[x_coords[cycle[:]]],y =biny[y_coords[cycle[:]]])#, hue = width_along_cycle, palette = 'magma')
                if len(cycle)>2:
                    for n in cycle:
                        if len(graph[n,:].indices) == 2:
                            for neig in graph[n,:].indices:
                                graph[n,neig] = 0
                                graph[neig,n] = 0
                            graph.eliminate_zeros()
                            break

                max_cycle, cycle= max_cycle_length(graph)
            
            # Exiting the while loop I either have found a true cycle or have nothing, in which case the cluster does not contain a true cycle:

            if not (np.histogram(data_a.statistic[x_coords[cycle[:]], y_coords[cycle[:]]], bins = 10, range = (0,360))[0] >= 1).all():
                print('just kidding, not a true cycle!')
                break
            
            len_along_cycle = data_len.statistic[x_coords[cycle[:]], y_coords[cycle[:]]]
            angle_along_cycle = data_a.statistic[x_coords[cycle[:]], y_coords[cycle[:]]]
            dtheta_along_cycle = data_dtheta.statistic[x_coords[cycle[:]], y_coords[cycle[:]]]
            width_along_cycle = binsize*distance[x_coords[cycle[:]], y_coords[cycle[:]]]
            
            # Compute and store some info about the cycle
            df.loc[len(df.index)]= [d.attributes['Timestep'],int(i),len(data),max_cycle*binsize, np.nanmean(width_along_cycle),np.nanmean(len_along_cycle),np.nanmean(dtheta_along_cycle),
                                     np.nanmean(binx[x_coords[cycle[:]]]),np.nanmean(biny[y_coords[cycle[:]]])]

    #computes if filaments at the edge of cluster are longer
    print('\t maxdist ',np.max(distlen_time))
    print('\t maxlen ',np.max(lens_time))
    if len(lens_time)> 2:

        distbinned_len=binned_statistic(np.array(distlen_time),np.array(lens_time), statistic=np.nanmean,bins = np.arange(41)+0.5)
        len_dist[f,:,0] +=distbinned_len.statistic
        distbinned_len_lquartile=binned_statistic(np.array(distlen_time),np.array(lens_time), statistic=lowerquart,bins = np.arange(41)+0.5)
        len_dist[f,:,1] +=distbinned_len_lquartile.statistic
        distbinned_len_uquartile=binned_statistic(np.array(distlen_time),np.array(lens_time), statistic=upperquart,bins = np.arange(41)+0.5)
        len_dist[f,:,2] +=distbinned_len_uquartile.statistic
    

    #print('frame ', f)
    print('\t num clusters ', nclust)
    #output = np.vstack([output,a])
    
#print(output.shape)   
np.savetxt('%s/polar_coarse_cluster_widths.txt'%(path),widths)
np.save('%s/polar_coarse_len_dist'%(path),len_dist)
#np.save('%s/polar_coarse_persistence'%(path),persistence)
df.to_csv(file_out,index = False)
outfile.close()
