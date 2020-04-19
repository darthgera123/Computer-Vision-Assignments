import numpy as np
# from PIL import Image
import cv2
import copy
from sklearn import mixture
# from math import sqrt
from igraph import Graph
from tqdm import tqdm

def getFgBgSet(im, bbox):
    pass

def getEdges(m1, m2, w, b, g, diag = False):
    edges = np.concatenate((m1, m2), axis = 1)
    w = w.reshape(-1,1)
    weights = g*np.exp(-b * w)
    if diag:
        weights = weights/np.sqrt(2)
    return edges, weights
    
def GrabCut(filename,outfile,mask, gamma,iterations=4,k = 5):
    
    im = cv2.imread(filename)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB) 
    
    h, w, _ = im.shape
    
    
    #Unassigned, Background sets
    _Tu = np.where(mask == 1)
    _Tb = np.where(mask == 0)
    
    #Fit GMMs for Background and Foreground
    gmmFg =  mixture.GaussianMixture(n_components = k)
    gmmBg =  mixture.GaussianMixture(n_components = k)
    
    
    #Defining neighbouring edge weights
    e_up = np.sum((im[1:,:] - im[:-1,:])**2,axis=2)
    e_left = np.sum((im[:,1:] - im[:,:-1])**2,axis=2)
    e_ul = np.sum((im[1:,1:] - im[:-1,:-1])**2,axis=2)
    e_ur = np.sum((im[1:,:-1] - im[:-1,1:])**2,axis=2)    

    #find beta
    b = 1/(2*(np.sum(e_up) + np.sum(e_left)+np.sum(e_ul) + np.sum(e_ur))/(4*h*w - 3*h - 3*w + 2))
    
#     gamma = 50
    
    img_indexes = np.arange(w*h).reshape(h,w)
    # Up edges and weights
    up_edges, up_weights = getEdges(img_indexes[1:, :].reshape(-1,1), img_indexes[:-1, :].reshape(-1,1), e_up, b, gamma)
    
    # Left edges and weights
    left_edges, left_weights = getEdges(img_indexes[:, 1:].reshape(-1,1), img_indexes[:, :-1].reshape(-1,1), e_left, b, gamma)
    
    # Up Left edges and weights
    ul_edges, ul_weights = getEdges(img_indexes[1:, 1:].reshape(-1,1), img_indexes[:-1, :-1].reshape(-1,1), e_ul, b, gamma, True)
    
    # Up Right edges and weights
    ur_edges, ur_weights = getEdges(img_indexes[1:, :-1].reshape(-1,1), img_indexes[:-1, 1:].reshape(-1,1), e_ur, b, gamma, True)
    
    #Concatenate all the images
    edges = np.concatenate((up_edges, left_edges, ul_edges, ur_edges), axis = 0)
    weights = np.concatenate((up_weights, left_weights, ul_weights, ur_weights), axis = 0)
   
    
    Tb = img_indexes[_Tb].reshape(-1,1)
    Tu = img_indexes[_Tu].reshape(-1,1)
    
    #Background edges to source with weights 0
    source_edges = np.concatenate((Tb, np.ones((Tb.shape[0],1))*(w*h)), axis = 1)
    source_weights = np.zeros((Tb.shape[0],1))

    #Background edges to terminal with high positive weight
    term_edges = np.concatenate((Tb, np.ones((Tb.shape[0],1))*(w*h + 1)), axis = 1)
    term_weights =  9 * gamma * np.ones((Tb.shape[0],1))
        
    edges = np.concatenate((edges, source_edges, term_edges), axis = 0)
    weights = np.concatenate((weights, source_weights, term_weights), axis = 0)
    
    Temp_fg = copy.deepcopy(Tu)
    Temp_bg = copy.deepcopy(Tb)
        
    im_temp = im.reshape([w*h,3])
    
    res_images = []
    
    for i in range(iterations):
        
        print("Running iteration : " + str(i+1))
        
        # re initialze the edges and weigths to originals
        temp_e = copy.deepcopy(edges)
        temp_w = copy.deepcopy(weights)
        
        # fit Fg and Bg pixels to corresponding GMMs
        gmmFg.fit(im_temp[Temp_fg].reshape(-1,3))
        gmmBg.fit(im_temp[Temp_bg].reshape(-1,3))
        
        # Find new weigths of Fg pixels to Source
        source_edges = np.concatenate((Tu, np.ones((Tu.shape[0],1))*(w*h)), axis = 1)
        source_weights = -gmmBg.score_samples(im_temp[Tu].reshape(-1,3)).reshape(-1,1)
        
        # Find new weigths of Fg pixels to Terminal
        term_edges = np.concatenate((Tu, np.ones((Tu.shape[0],1))*(w*h + 1)), axis = 1)
        term_weights = -gmmFg.score_samples(im_temp[Tu].reshape(-1,3)).reshape(-1,1)
        
        # final edge and weigth array
        temp_e = np.concatenate((temp_e, source_edges, term_edges), axis = 0)
        temp_w = np.concatenate((temp_w, source_weights, term_weights), axis = 0)
        temp_e = temp_e.astype(int)
        
        # Graph cut
        cuts = Graph()
        cuts.es['weight'] = 1
        cuts.add_vertices(h * w + 2)
        cuts.add_edges(temp_e)
        cuts.es['weight'] = temp_w.flatten()

        c = cuts.mincut(w * h, w * h + 1, capacity='weight')
        
        # new Fg and Bg indices
        indexb = np.array(list(c[1]),dtype=int)
        indexb = indexb[indexb < w*h]
        indexf = np.array(list(c[0]),dtype=int)
        indexf = indexf[indexf < w*h]
        
        Temp_fg = indexf
        Temp_bg = indexb
        t = copy.deepcopy(im_temp)
        t[Temp_bg] = [0,0,0]
        res_images.append(t.reshape([h,w,3]))
        
    return res_images[-1]