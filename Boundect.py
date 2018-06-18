# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:27:04 2018

@author: sfarooq1
"""

import numpy as np
import Centroid
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection as Poly3D
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import math
import random
import sys
from matplotlib.patches import Circle
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

def import_data(fname = '3lzg.pdb'):
    with open(fname, 'r') as f:
        coordinates,residueid,residue = [],np.array([]),np.array([]),
        atom = np.array([])
        for row in f:
            line = row.split()
            if line[0] == 'ATOM':
                atom = np.append(atom,line[-1])
                residue = np.append(residue, line[3])
                fl = [float(num) for num in line[6:9]]
                coordinates.append(fl)
                residueid = np.append(residueid, line[5])
    return np.array(coordinates),residueid,residue,atom

def plot_atoms(coordinates, atom):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dclr = {'C':'k','N':'b','O':'r','S':'y'}
    clr = [dclr[atm] for atm in atom]
    ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2],c=clr,
               marker='.')
    plt.show()
    
def atomic_center(coords, mean = True, include_radius = True):
    if len(coords) == 1:
        if include_radius:
            return coords[0],0
        return coords[0]
    elif len(coords) == 2:
        mean = True
    if mean:
        center = np.mean(coords, axis=0)
        if include_radius:
            radius = 0
            for c in coords:
                d = euc(c,center)
                if d > radius: radius = d
            return center,radius
        return center
    center,radius = Centroid.warm_start_bubble(np.array(coords))
    if include_radius:
        return center,radius
    return center

def avg_residues(coordinates, residueid, mean = True, plt_show = False):
    prev_res_id,coords,residue_coords = residueid[0],[],[]
    residue_radius = np.array([])
    i = 0
    while i < len(coordinates):
        cur_res_id = residueid[i]
        if cur_res_id != prev_res_id:
            cntr,mxdist = atomic_center(coords, mean)
            residue_coords.append(cntr)
            residue_radius = np.append(residue_radius, mxdist)
            coords = []
            prev_res_id = cur_res_id
        else:
            coords.append(coordinates[i])
            i += 1
    cntr,mxdist = atomic_center(coords, mean)
    residue_coords.append(cntr)
    residue_radius = np.append(residue_radius, mxdist)
    residue_coords = np.array(residue_coords)
    if not plt_show:
        return residue_coords,residue_radius
    tri = Delaunay(residue_coords)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    u = np.unique(tri.convex_hull)
    ax.scatter(residue_coords[u,0],residue_coords[u,1],residue_coords[u,2],
               marker='.')
    plt.show()
    return residue_coords,residue_radius

def plot_2D_with_radius(coords,radius,plane='xy',proportion=1):
    if plane == 'xy':
        coord2 = coords[:,[0,1]]
    elif plane == 'yz':
        coord2 = coords[:,[1,2]]
    else:
        coord2 = coords[:,[0,2]]
    if proportion < 1:
        n = int(len(coord2)*proportion)
        samples = random.sample(range(len(coord2)),n)
        coord2 = coord2[samples]
        radius = radius[samples]
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.scatter(coord2[:,0],coord2[:,1])
    for i in range(len(radius)):
        ax.add_patch(Circle((coord2[i,0],coord2[i,1]),radius[i],fc='none',
                            ec='green',fill=False))
    plt.show()

def euc(x,y):
    return np.sqrt(np.sum((x-y)**2))

def simplices_content(points):
    if len(points) == 2:
        return np.sqrt(np.sum((points[0]-points[1])**2))
    d = len(points[0])
    M = np.zeros((d,d))
    v0 = points[0]
    for i in range(1,len(points)):
        M[:,i-1] = points[i] - v0
    return np.abs((1/math.factorial(d))*np.linalg.det(M))

def s2f(num, str_to_fit):
    s = str(num)
    return '0'*(len(str_to_fit)-len(s))+s

def plot_simplex(ax, vertices, surface_color = 'b', edge_color = 'same'):
    if len(vertices) == 3:
        poly = Poly3D([vertices])
        poly.set_color(surface_color)
        if (type(edge_color) is str) and (edge_color == 'same'):
            edge_color = surface_color
        poly.set_edgecolor(edge_color)
        ax.add_collection3d(poly)
        return poly
    else:
        I = np.arange(len(vertices))
        for i in range(len(vertices)):
            poly = Poly3D([vertices[I != i]])
            poly.set_color(surface_color)
            if (type(edge_color) is str) and (edge_color == 'same'):
                edge_color = surface_color
            poly.set_edgecolor(edge_color)
            ax.add_collection3d(poly)

def concat(points, mx_len):
    p = np.sort(points)
    s = ''
    for i in p:
        k = str(i)
        s += '0'*(mx_len-len(k))+k
    return s

def parseback(s, mx_len):
    bi,ai,points = 0,mx_len,np.array([],dtype=int)
    for i in range(int(len(s)/mx_len)):
        points = np.append(points, int(s[bi:ai]))
        bi = ai
        ai += mx_len
    return points

def simplices_enclosed(centers, radii):
    intersects = True
    for i in range(len(centers)):
        for j in range(i+1,len(centers)):
            V = centers[j] - centers[i]
            if not np.any(V): # if one of the centers are the same: raise error
                raise AssertionError("Cannot have circles with same center!")
            V = V / np.linalg.norm(V)
            x0 = centers[j] + radii[j]*V
            x1 = centers[j] - radii[j]*V
            F = min([euc(x0,centers[i]),euc(x1,centers[i])]) <= radii[i]
            V = centers[i] - centers[j]
            V = V / np.linalg.norm(V)
            x0 = centers[i] + radii[i]*V
            x1 = centers[i] - radii[i]*V
            S = min([euc(x0,centers[j]),euc(x1,centers[j])]) <= radii[j]
            if F and S:
                continue # Intersection
            elif F or S:
                intersects = 'Overlap' # Overlapping routine needs to fire
            else:
                return False # Meaning it is not enclosed, so delete simplice
    return intersects

def in_hull(p, hull):
    if len(hull) <= len(p): return False
    hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def quadratic_eq(a,b,c):
    det = b**2-4*a*c
    if det < 0:
        raise AssertionError("Negative determinant!")
    det = np.sqrt(det)
    return (-b+det)/(2*a),(-b-det)/(2*a)

def new_point_bulges(centers, radii, new_center, new_radius,allow_bulge = True,
                     ax = False):
    intersects = simplices_enclosed(centers, radii)
    if not intersects:
        return True # If there is no intersection then it automatically bulges.
    if not allow_bulge and (len(new_center) < 3):
        return False # If bulging is not allowed then no need to calculate.
    if intersects == 'Overlap':
        return False # Assume overlapping implies conjestion. Or comment out.
        # Alternatively (if above is commented out):
        r = np.max(radii) # If one of them overlapped, assume all radii are big
        for i in range(len(radii)): # This makes calculation easier, although
            radii[i] = r # potentially not as accurate.
    A = np.zeros((len(centers)-1,len(centers[0])))
    b = np.zeros((len(centers)-1,))
    for i in range(1,len(centers)): # Intersecting spheres -> yields hyperplane
        A[i-1,:] = 2*centers[i] - 2*centers[0]
        b[i-1] = np.sum(centers[i]**2) - np.sum(centers[0]**2)
        b[i-1] += radii[0]**2 - radii[i]**2
    v = np.zeros((len(centers[0]),))
    for i in range(len(v)): # Intersecting hyperplanes -> yields a line.
        v[i] = (-1**i)*np.linalg.det(np.delete(A,i,axis=1)) #This is its vector
    p = np.linalg.solve(A[:,:-1], b) # Find a point on the line of intersection
    p = np.append(p, 0) # Because last dimension was set to 0.
    if len(new_center) >= 3: # if >2 dim then check where the line is.
        V = np.zeros((len(centers)-1,len(centers[0])))
        for i in range(1,len(centers)):
            V[i-1,:] = centers[i] - centers[0]
        plane = np.zeros((len(centers[0]),))
        for i in range(len(plane)): # Find the equation of the plane of centers
            plane[i] = (-1**i)*np.linalg.det(np.delete(V,i,axis=1))
        D = np.sum(plane*centers[0])
        t = (D - np.sum(plane*p))/(np.sum(v*plane))
        m = p + v*t # The theoretical middle of the centers
        if not in_hull(m, centers): #Checking the position of intersection line
            return False # Assume False b/c this condition not currently suprtd
        all_greater = True
        for i in range(len(centers)):
            if euc(centers[i],m) <= radii[i]:
                all_greater = False
        if all_greater:
            return True # This means there is a gap inbetween the circles!
        elif not allow_bulge:
            return False # No hole in the middle and we do not allow bulge.
    else:
        v[1] = -v[1] # Because there are no intersections with 1 line!
    a = np.sum(v**2) # Now we solve intersection of line with new sphere
    b = 2*np.sum(p*v) - 2*np.sum(v*new_center)
    c = np.sum(p**2)-2*np.sum(p*new_center)+np.sum(new_center**2)-new_radius**2
    try:
        t1,t2 = quadratic_eq(a,b,c)
    except AssertionError:
        return False # This means the line never intersects!
    i1,i2 = p + v*t1, p + v*t2 # Line intersects sphere in two places.
    i=np.argmin([euc(i1,centers[0]),euc(i2,centers[0])])#pick closest to cnters
    intersection = i2 if i else i1
    for i in range(len(centers)):
        if euc(centers[i],intersection) <= radii[i]:
            return False # Theoretical bulge point is covered by a sphere.
    return True # Theoretical bulge point is uncovered!

def abline(ax, slope, intercept):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    ylim = ax.get_ylim()
    y_vals = intercept + slope * x_vals
    if y_vals[0] > ylim[1]:
        y_vals[0] = ylim[1]
        x_vals[0] = (ylim[1]-intercept)/slope
    elif y_vals[0] < ylim[0]:
        y_vals[0] = ylim[0]
        x_vals[0] = (ylim[0]-intercept)/slope
    if y_vals[1] > ylim[1]:
        y_vals[1] = ylim[1]
        x_vals[1] = (ylim[1]-intercept)/slope
    elif y_vals[1] < ylim[0]:
        y_vals[1] = ylim[0]
        x_vals[1] = (ylim[0]-intercept)/slope
    ax.plot(x_vals, y_vals, 'b--')

def bulgeplot():
    c = np.random.uniform(size=(2,2))
    c2 = np.random.uniform(size=2)
    r = np.random.uniform(0.25,size=2)
    r2 = np.random.uniform(0.25)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(c[:,0],c[:,1],marker='+',color='b')
    ax.add_patch(Circle((c[0,0],c[0,1]),r[0],fc=None,ec='b',fill=False))
    ax.add_patch(Circle((c[1,0],c[1,1]),r[1],fc=None,ec='b',fill=False))
    xy,y = 2*c[1] - 2*c[0],np.sum(c[1]**2)-np.sum(c[0]**2)+r[0]**2-r[1]**2
    ax.scatter(c2[0],c2[1],marker='+',color='r')
    ax.add_patch(Circle((c2[0],c2[1]),r2,fc=None,ec='r',fill=False))
    abline(ax, -xy[0]/xy[1], y/xy[1])
    intersect = simplices_enclosed(c,r)
    print("Intersect: ",intersect)
    print("Bulges: ",new_point_bulges(c,r,c2,r2,True,ax))
    plt.show()

class Node:
    def __init__(self, border_code, increased_revealage):
        self.code = border_code
        self.increased_revealage = increased_revealage
        self.child_node = None
        self.parent_node = None
        
class AutoDescSort_Queue:
    def __init__(self):
        self.head = None
        self.length = 0
    def add(self, border_code, increased_revealage):
        self.length += 1
        new_node = Node(border_code, increased_revealage)
        if self.head == None:
            self.head = new_node
        else:
            cur_node = self.head
            finished_update = False
            while cur_node.increased_revealage > increased_revealage:
                if cur_node.child_node == None:
                    cur_node.child_node = new_node
                    new_node.parent_node = cur_node
                    finished_update = True
                    break
                else:
                    cur_node = cur_node.child_node
            if not finished_update:
                if cur_node.parent_node == None:
                    new_node.child_node = cur_node
                    cur_node.parent_node = new_node
                    self.head = new_node
                else:
                    new_node.parent_node = cur_node.parent_node
                    new_node.child_node = cur_node
                    cur_node.parent_node.child_node = new_node
                    cur_node.parent_node = new_node
    def pop(self):
        if self.head == None:
            raise AssertionError("Nothing Left to Pop!")
        self.length -= 1
        code,increased_revealage = self.head.code,self.head.increased_revealage
        self.head.child_node.parent_node = None
        self.head = self.head.child_node
        return code,increased_revealage
    def disp(self):
        cur_node = self.head
        print("Auto Descending Sort Queue of Length: "+str(self.length))
        while cur_node != None:
            print(cur_node.code, cur_node.increased_revealage)
            cur_node = cur_node.child_node

class ConcaveHull:
    def __init__(self, coordinates):
        self.tri = Delaunay(coordinates)
        self.tri_code_mx_len = len(str(self.tri.npoints-1))
        self.I = np.arange(len(self.tri.simplices[0]))
        self.activated = False
    def add_simplex(self, simplex_id):
        simplex_points = self.tri.simplices[simplex_id]
        content = simplices_content(self.tri.points[simplex_points])
        self.total_content += content
        self.simplex_content[simplex_id] = content
        for i in range(len(self.I)):
            plane_points = simplex_points[self.I != i]
            plane_str_code = concat(plane_points, self.tri_code_mx_len)
            try:
                self.planes[plane_str_code].add(simplex_id)
                new_amt = len(self.planes[plane_str_code])
            except KeyError:
                self.planes[plane_str_code] = {simplex_id}
                new_amt = 1
            if new_amt == 2:
                self.borders.pop(plane_str_code)
            elif new_amt == 1:
                self.borders[plane_str_code] = plane_points
    def remove_simplex(self, simplex_id):
        simplex_points = self.tri.simplices[simplex_id]
        self.total_content -= self.simplex_content[simplex_id]
        for i in range(len(self.I)):
            plane_points = simplex_points[self.I != i]
            code = concat(plane_points, self.tri_code_mx_len)
            self.planes[code].remove(simplex_id)
            new_amt = len(self.planes[code])
            if new_amt == 1:
                new_SA = simplices_content(self.tri.points[plane_points])
                self.total_surface_area += new_SA
                self.surface_area[code] = new_SA
                self.borders[code] = plane_points
                self.Queue.add(code, new_SA)
            elif new_amt == 0:
                self.popped_codes.append(code)
                self.borders.pop(code)
                self.total_surface_area -= self.surface_area.pop(code)
        self.popped_simplexes.append(simplex_id)
        for point in self.point_count:
            try:
                self.border_points[point] += self.point_count[point]
            except KeyError:
                self.border_points[point] = self.point_count[point]
    def get_simplex_neighbors(self):
        self.activated = True
        self.planes,self.borders,self.total_content = {},{},0
        self.simplex_content = np.zeros((len(self.tri.simplices),))
        for i in range(len(self.tri.simplices)):
            self.add_simplex(i)
        self.surface_area,self.total_surface_area,self.mx_touches = {},0,0
        self.border_points = {}
        self.Queue = AutoDescSort_Queue()
        self.popped_codes,self.popped_simplexes = [],[]
        for code,border in self.borders.items():
            surface_area = simplices_content(self.tri.points[border])
            self.total_surface_area += surface_area
            self.surface_area[code] = surface_area
            self.Queue.add(code, surface_area)
            for b in border:
                try:
                    current = self.border_points[b] + 1
                    self.border_points[b] = current
                    if current > self.mx_touches:
                        self.mx_touches = current
                except KeyError:
                    self.border_points[b] = 1
    def determine_point_count(self, simplex_id):
        simplex_points = self.tri.simplices[simplex_id]
        self.point_count = {}
        for i in range(len(self.I)):
            plane_points = simplex_points[self.I != i]
            code = concat(plane_points, self.tri_code_mx_len)
            if code in self.borders:
                for point in plane_points:
                    try:
                        self.point_count[point] -= 1
                    except KeyError:
                        self.point_count[point] = -1
            else:
                for point in plane_points:
                    try:
                        self.point_count[point] += 1
                    except KeyError:
                        self.point_count[point] = 1
    def find_largest_border_simplex(self, allow_pinch = False, 
            allow_puncture = False,radially_restricted = False,
            allow_bulge = True,return_border_code = False):
        check_not_passed = True
        while check_not_passed and (self.Queue.length > 0):
            border_code, inc_SA = self.Queue.pop()
            try:
                sid = list(self.planes[border_code])[0]
            except IndexError:
                continue
            self.determine_point_count(sid)
            if (not allow_pinch) or (not allow_puncture):
                punctured,pinched = False,False
                for point in self.point_count:
                    if point not in self.border_points:
                        continue
                    new_total=self.border_points[point]+self.point_count[point]
                    if new_total <= 0:
                        punctured = True
                    elif new_total > self.mx_touches:
                        pinched = True
                if not allow_puncture and not allow_pinch:
                    if not punctured and not pinched:
                        check_not_passed = False
                elif not allow_puncture:
                    if not punctured:
                        check_not_passed = False
                elif not allow_pinch:
                    if not pinched:
                        check_not_passed = False
            else:
                check_not_passed = False
            if not check_not_passed and np.all(radially_restricted):
                centers = self.tri.points[self.borders[border_code]]
                radii = radially_restricted[self.borders[border_code]]
                npt=set(self.tri.simplices[sid])-set(self.borders[border_code])
                npt = npt.pop()
                nc,nr=self.tri.points[npt],radially_restricted[npt]
                check_not_passed = not new_point_bulges(centers,radii,nc,nr,
                                                        allow_bulge)
        if check_not_passed:
            return None
        if return_border_code:
            return sid,border_code
        return sid
    def shrink_wrap(self,allow_pinch = False, allow_puncture = False, 
                    radially_restricted = False, allow_bulge = True,
                    verbose = False):
        if not self.activated:
            self.get_simplex_neighbors()
        removable = True
        mQl = len(str(len(self.tri.simplices)))
        while removable:
            if verbose:
                Ql = str(self.Queue.length)
                print("\rQueue Size: "+'0'*(mQl-len(Ql))+Ql,end='')
                sys.stdout.flush()
            try:
                sid=self.find_largest_border_simplex(allow_pinch,
                    allow_puncture,radially_restricted,allow_bulge)
            except AttributeError:
                sid = None
            if sid != None:
                self.remove_simplex(sid)
            else:
                removable = False
        if verbose:
            Ql = str(self.Queue.length)
            print("\rQueue Size: "+'0'*(mQl-len(Ql))+Ql,end='')
            sys.stdout.flush()
            print("")
    def get_border_points(self):
        point_set = set()
        for code,points in self.borders:
            for p in points:
                point_set.add(p)
        return self.tri.points[list(point_set)]
    def rank_simplices(self):
        self.simpli_rank = np.array([])
        self.simpli_content = np.array([])
        for simpli in self.tri.simplices:
            dists = []
            content = simplices_content(self.tri.points[simpli])
            for i in range(len(simpli)):
                for j in range(i+1,len(simpli)):
                    dists.append(self.dist[simpli[i],simpli[j]])
            self.simpli_rank = np.append(self.simpli_rank, np.max(dists))
            self.simpli_content = np.append(self.simpli_content, content)
    def find_optimal_content(self, plot_curve = False):
        self.argsorted_sr = np.argsort(self.simpli_rank)[::-1]
        s = np.sum(self.simpli_content)
        C = np.array([s])
        for i in range(len(self.argsorted_sr)):
            s -= self.asimpli_content[self.argsorted_sr[i]]
            C = np.append(C, s)
        X = range(len(C))
        if plot_curve:
            fig = plt.figure(figsize=(15.5,8.5))
            ax = fig.add_subplot(111)
            ax.scatter(X,C)
            plt.show()
    
def plot_hull(tri):
    fig = plt.figure(figsize=(15.5,8.5))
    ax = fig.add_subplot(111, projection='3d')
    red = np.linspace(0,1,len(tri.convex_hull))
    xmin,xmax = min(tri.points[:,0]),max(tri.points[:,0])
    ymin,ymax = min(tri.points[:,1]),max(tri.points[:,1])
    zmin,zmax = min(tri.points[:,1]),max(tri.points[:,2])
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.set_zlim([zmin,zmax])
    for i in range(len(tri.convex_hull)):
        plot_simplex(ax,tri.points[tri.convex_hull[i]],[red[i],104/255,54/255])
        plt.pause(0.5)
    plt.show()
    
def plot_2D(points, radii, allow_pinch = False, allow_puncture = False,
            specify_start = False, plot_radii = False):
    fig = plt.figure(figsize=(15.5,8.5))
    ax = fig.add_subplot(121)
    ch = ConcaveHull(points)
    ax.scatter(ch.tri.points[:,0],ch.tri.points[:,1])
    if plot_radii:
        for i in range(len(radii)):
            ax.add_patch(Circle(tuple(points[i]),radii[i],fc='none',fill=False,
                                ec='g'))
    ch.get_simplex_neighbors()
    lines = []
    for borders in list(ch.borders.values()):
        b = ch.tri.points[borders]
        line, = ax.plot(b[:,0],b[:,1],color='orange')
        lines.append(line)
    ax2 = fig.add_subplot(122)
    ax2.scatter(0,ch.total_content)
    ax2.set_xlim(right=len(ch.tri.simplices))
    ax2.set_ylim(bottom=0)
    instance = 1
    if specify_start:
        input("Press [Enter] to continue.")
    try:
        while len(ch.borders) > 0:
            plt.pause(0.2)
            a=ch.find_largest_border_simplex(allow_pinch,allow_puncture,radii)
            L = len(lines)
            for i in range(L):
                line = lines.pop()
                ax.lines.remove(line)
            ch.remove_simplex(a)
            for borders in list(ch.borders.values()):
                b = ch.tri.points[borders]
                line, = ax.plot(b[:,0],b[:,1],color='orange')
                lines.append(line)
            ax2.scatter(instance,ch.total_content,color='b')
            instance += 1
    except AssertionError:
        return ch
    
def before_and_after_2D(points, radii, allow_bulge = True):
    fig = plt.figure(figsize=(15.5,8.5))
    ax = fig.add_subplot(121)
    ch = ConcaveHull(points)
    ax.scatter(ch.tri.points[:,0],ch.tri.points[:,1])
    ch.get_simplex_neighbors()
    for borders in list(ch.borders.values()):
        b = ch.tri.points[borders]
        ax.plot(b[:,0],b[:,1],color='orange')
    border_points = points[list(ch.border_points.keys())]
    ax.scatter(border_points[:,0],border_points[:,1],color='r')
    ax.set_title('Convex Hull', fontsize=16)
    ch.shrink_wrap(False, False, radii, allow_bulge, True)
    ax = fig.add_subplot(122)
    ax.scatter(ch.tri.points[:,0],ch.tri.points[:,1])
    for borders in list(ch.borders.values()):
        b = ch.tri.points[borders]
        ax.plot(b[:,0],b[:,1],color='orange')
    border_points = points[list(ch.border_points.keys())]
    ax.scatter(border_points[:,0],border_points[:,1],color='r')
    ax.set_title('Concave Hull', fontsize=16)
    plt.show()
        
def plot_3D(rcoord, radii, pause = 0.01):
    fig = plt.figure(figsize=(15.5,8.5))
    ax = fig.add_subplot(121, projection='3d')
    sct = ax.scatter(rcoord[:,0],rcoord[:,1],rcoord[:,2])
    sct.remove()
    mean = np.mean(rcoord,axis=0)
    farthest_distance = 0
    for i in range(len(rcoord)):
        d = euc(rcoord[i],mean)
        if d > farthest_distance:
            farthest_distance = d
    ch = ConcaveHull(rcoord)
    ch.get_simplex_neighbors()
    before = set(ch.borders.keys())
    Polys = {}
    for code,border in ch.borders.items():
        points = ch.tri.points[border]
        red = 1 - (euc(np.mean(points,axis=0),mean)/farthest_distance)
        Polys[code] = plot_simplex(ax,points,[red,104/255,54/255],'k')
    ax2 = fig.add_subplot(122)
    ax2.scatter(0,ch.total_content)
    instance = 1
    ax2.set_xlim(right = len(ch.tri.simplices))
    ax2.set_ylim(bottom = 0)
    while len(ch.borders) > 0:
        plt.pause(pause)
        a,code = ch.find_largest_border_simplex(False,False,radii,True)
        poly = Polys.pop(code)
        poly.remove()
        ch.remove_simplex(a)
        after = set(ch.borders.keys())
        new_codes = after - before
        for code in new_codes:
            points = ch.tri.points[ch.borders[code]]
            red = 1 - (euc(np.mean(points,axis=0),mean)/farthest_distance)
            Polys[code] = plot_simplex(ax,points,[red,104/255,54/255],'k')
        ax2.scatter(instance, ch.total_content, color = 'b')
        instance += 1
        before = after
        
def before_and_after_3D(points, radii, allow_bulge = True, beta = 1):
    fig = plt.figure(figsize=(15.5,8.5))
    ax = fig.add_subplot(121, projection = '3d')
    ch = ConcaveHull(points)
    ch.get_simplex_neighbors()
    t = np.zeros((len(points),),dtype=bool)
    bp = list(ch.border_points.keys())
    t[bp] = True
    b_points = points[t]
    ax.scatter(b_points[:,0],b_points[:,1],b_points[:,2],color='r', alpha=0.5)
    t = np.ones((len(points),),dtype=bool)
    t[bp] = False
    i_points = points[t]
    ax.scatter(i_points[:,0],i_points[:,1],i_points[:,2],color='b')
    ax.set_title('Convex Hull', fontsize=16)
    ch.shrink_wrap(False, False, beta*radii, allow_bulge, True)
    ax =fig.add_subplot(122,projection = '3d')
    t = np.zeros((len(points),),dtype=bool)
    bp = list(ch.border_points.keys())
    t[bp] = True
    b_points = points[t]
    ax.scatter(b_points[:,0],b_points[:,1],b_points[:,2],color='r', alpha=0.5)
    t = np.ones((len(points),),dtype=bool)
    t[bp] = False
    i_points = points[t]
    ax.scatter(i_points[:,0],i_points[:,1],i_points[:,2],color='b')
    ax.set_title(r'Concave Hull ($\beta = '+str(beta)+r'$)', fontsize=16)
    print(len(b_points),len(i_points))
    plt.show()
    
def beta_parameter_3D(points, radii, allow_bulge = False, steps = 60):
    ch = ConcaveHull(points)
    beta = np.linspace(1,3,steps)
    ratio,step = [],1
    for b in beta:
        s = str(steps)
        print("\rStep: "+'0'*(len(s)-len(str(step)))+str(step)+" / "+s,end='')
        sys.stdout.flush()
        step += 1
        ch.get_simplex_neighbors()
        if step == 2:
            cvx = len(ch.border_points)/float(len(points))
        ch.shrink_wrap(False,False,b*radii,allow_bulge)
        ratio.append(len(ch.border_points)/float(len(points)))
    print("")
    fig = plt.figure(figsize=(15.5,8.5))
    ax = fig.add_subplot(111)
    ax.plot(beta, ratio)
    ax.plot([1,3],[cvx, cvx],'--',color='orange')
    ax.text(1.05, cvx+0.05, 'Convex Hull',color='orange')
    ax.set_ylim(bottom=0,top=1)
    ax.set_ylabel("# of Border Points / # of Total Points", fontsize=14)
    ax.set_xlabel("Beta parameter (multiplying radius)", fontsize=14)
    ax.set_title("Finding ideal beta for 3D Concave Hull",fontsize=16)
    plt.show()
        