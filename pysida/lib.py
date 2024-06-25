from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta, datetime
import glob
from multiprocessing import Pool
import os

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pynextsim.nextsim_mesh import NextsimMesh
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.optimize import curve_fit

DAY_SECONDS = 24 * 60 * 60
DIST2COAST_NC = None


class BaseRunner:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)


@dataclass
class Pair:
    x0: np.ndarray
    x1: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    d0: pd.Timestamp
    d1: pd.Timestamp
    t: np.ndarray
    a: np.ndarray
    p: np.ndarray
    g: np.ndarray

    def triplot(self):
        plt.triplot(self.x0, self.y0, self.t, mask=~self.g)

    def get_speed(self):
        time_delta = (self.d1 - self.d0).total_seconds()
        u = (self.x1 - self.x0) / time_delta
        v = (self.y1 - self.y0) / time_delta
        return u, v

    def get_velocity_gradients(self, u, v):
        xt, yt, ut, vt = [i[self.t].T for i in (self.x0, self.y0, u, v)]
        ux, uy = get_velocity_gradient_elems(xt, yt, ut, self.a)
        vx, vy = get_velocity_gradient_elems(xt, yt, vt, self.a)
        return ux, uy, vx, vy


class MeshFileList:
    def __init__(self, idir, mask='mesh_[0-9]*bin', lazy=True):
        self.files = sorted(glob.glob(f'{idir}/{mask}'))
        self.dates = [datetime.strptime(os.path.basename(f).split('_')[1].split('Z')[0], '%Y%m%dT%H%M%S') for f in self.files]
        self.data = {}
        if not lazy:
            self.read_all_data()

    def read_all_data(self):
        print(f'Reading data from {self.dates[0]} to {self.dates[-1]}')
        for idate in self.dates:
            self.read_one_file(idate)

    def read_one_file(self, idate):
        ifile = self.files[self.dates.index(idate)]
        m = NextsimMesh(ifile)
        self.data[idate] = dict(
            x = m.nodes_x,
            y = m.nodes_y,
            i = m.get_var('id'),
        )

    def find_nearest(self, date):
        date_diff = np.abs(np.array(self.dates) - np.array(date))
        i = np.argmin(date_diff)
        return self.files[i], self.dates[i]

    def get_data(self, idate):
        if idate not in self.data:
            self.read_one_file(idate)
        return self.data[idate]['x'], self.data[idate]['y'], self.data[idate]['i']


@dataclass
class Defor:
    e1: np.ndarray
    e2: np.ndarray
    e3: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    vx: np.ndarray
    vy: np.ndarray

    def tripcolor(self, p, vmin=0, vmax=0.1, name='e0', units='d', cmap='plasma_r'):
        v = np.array(self.__dict__[name])
        if units == 'd':
            v *= DAY_SECONDS
        plt.tripcolor(p.x0, p.y0, v, triangles=p.t, mask=~p.g, vmin=vmin, vmax=vmax, cmap=cmap)


@dataclass
class MarsanPair:
    x: np.ndarray
    y: np.ndarray
    a: np.ndarray
    ux: np.ndarray
    vx: np.ndarray
    uy: np.ndarray
    vy: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    e3: np.ndarray


class PairFilter:
    # minimum number of days between image pair
    # lower value allows more matching pairs but increases the tail of PDF
    min_time_diff = 2.5
    # maximum number of days between image pair
    # higher value allows more matching images but increases range of values
    max_time_diff = 4.5
    # aggregation window (days)
    window = '3D'
    # minimum area of triangle element to select for computing deformation
    # higher value decrease the number of elements but may exclude some
    # errorneous triangles appearing between RS2 swaths
    min_area_resolution = {
        5000  :   8e6,
        10000 :  20e6,
        20000 : 120e6,
    }
    # maximum area of triangle element to select for computing deformation
    # lower value decrease the number of elements but may exclude some
    # errorneous triangles appearing between RS2 swaths
    max_area_resolution = {
        5000  :  20e6,
        10000 :  80e6,
        20000 : 320e6,
    }
    # minimum area/perimeter ratio to select for computing deformation
    # lower value allow sharper angles in triangles and increases number of elements
    min_ap_ratio = 0.17
    # minimum distance from coast
    min_dist=100
    # minimum displacement between nodes
    min_drift=1000
    # minimum size of triangulation
    # lower number increases number of triangles but increase chance of noise
    min_tri_size = 10
    # path to the dist2coas NPY file
    dist2coast_path = None
    dist2coast_path = None

    def __init__(self, pairs, defor, resolution, **kwargs):
        self.min_area = self.min_area_resolution[resolution]
        self.max_area = self.max_area_resolution[resolution]
        self.__dict__.update(kwargs)
        self.pdefor_src = self.merge_pairs_defor(pairs, defor)
        if self.dist2coast_path is not None:
            self.dist2coast = np.load(self.dist2coast_path)

    def merge_pairs_defor(self, pairs, defor):
        pdefor = []
        for p, d in zip(pairs, defor):
            if p is None:
                continue
            p.__dict__.update(d.__dict__)
            pdefor.append(p)
        return pdefor

    def filter(self, date):
        pdefor_dst = []
        for p in self.pdefor_src:
            time_diff = (p.d1 - p.d0).total_seconds() / DAY_SECONDS
            if (
                (self.min_time_diff <= time_diff) and
                (self.max_time_diff >= time_diff) and
                (date <= p.d0 < (date + pd.Timedelta(self.window)))
                ):
                gpi = (
                    (p.a >= self.min_area) *
                    (p.a <= self.max_area) *
                    (np.sqrt(p.a)/p.p >= self.min_ap_ratio)
                )
                if self.min_dist > 0 and self.dist2coast is not None:
                    xel = p.x0[p.t].mean(axis=1)
                    yel = p.y0[p.t].mean(axis=1)
                    dist = self.get_dist2coast(xel, yel)
                    gpi *= (dist > self.min_dist)

                if self.min_drift > 0:
                    drift = np.hypot(p.x1 - p.x0, p.y1 - p.y0)[p.t].mean(axis=1)
                    gpi *= (drift > self.min_drift)

                gpi = np.where(gpi)[0]
                if gpi.size > self.min_tri_size:
                    p.g[:] = False
                    p.g[gpi] = True
                    pdefor_dst.append(p)

        return pdefor_dst

    def get_dist2coast(self, x, y, srs_dst=None):
        if srs_dst is None:
            srs_dst = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=60)
        dist = np.zeros_like(x) + np.nan
        srs_wgs = ccrs.PlateCarree()
        lon, lat, _ = srs_wgs.transform_points(srs_dst, x, y).T
        col = np.round((180 + lon)*25).astype(int)
        row = np.round((90 - lat)*25).astype(int)
        gpi = (col > 0) * (col < self.dist2coast.shape[1]) * (row > 0) * (row < self.dist2coast.shape[0])
        dist[gpi] = self.dist2coast[row[gpi], col[gpi]]
        return dist


class MarsanSpatialScaling:
    scale0_resolution = {
        5000  : 3500,
        10000 : 7000,
        20000 : 14000,
    }
    scales_resolution = {
        5000  : (7e3, 14e3, 28e3, 56e3, 112e3, 224e3, 448e3, 896e3),
        10000 :      (14e3, 28e3, 56e3, 112e3, 224e3, 448e3, 896e3),
        20000 :            (28e3, 56e3, 112e3, 224e3, 448e3, 896e3),
    }
    resolution = 10000

    def __init__(self, pf, resolution):
        self.pf = pf
        self.resolution = resolution

    def merge_pairs(self, pdefor):
        m = defaultdict(list)
        for p in pdefor:
            m['x'].append(p.x0[p.t].mean(axis=1)[p.g])
            m['y'].append(p.y0[p.t].mean(axis=1)[p.g])
            m['a'].append(p.a[p.g])
            m['ux'].append(p.ux[p.g])
            m['uy'].append(p.uy[p.g])
            m['vx'].append(p.vx[p.g])
            m['vy'].append(p.vy[p.g])
            m['e1'].append(p.e1[p.g])
            m['e2'].append(p.e2[p.g])
            m['e3'].append(p.e3[p.g])

        for key in m:
            m[key] = np.hstack(m[key])
        return MarsanPair(**m)

    def coarse_grain(self, p):
        vel_grad_names = ['ux', 'uy', 'vx', 'vy']
        result = defaultdict(dict)
        scales = self.scales_resolution[self.resolution]
        scale0 = self.scale0_resolution[self.resolution]
        for scale in scales:
            x_bins = np.arange(p.x.min(), p.x.max(), scale)
            y_bins = np.arange(p.y.min(), p.y.max(), scale)

            grids = {i:np.zeros((y_bins.size, x_bins.size)) + np.nan
                    for i in ['a']+vel_grad_names}

            for row, yb0 in enumerate(y_bins):
                for col, xb0 in enumerate(x_bins):
                    gpi = np.where(
                        (p.x >= xb0) *
                        (p.y >= yb0) *
                        (p.x < (xb0 + scale)) *
                        (p.y < (yb0 + scale)) *
                        (p.a < (scale*1.5)**2)
                    )[0]
                    if gpi.size > 0:
                        grids['a'][row, col] = p.a[gpi].sum()
                        for i in vel_grad_names:
                            grids[i][row, col] = (p.__dict__[i][gpi] * p.a[gpi]).sum() / p.a[gpi].sum()

            gpi = np.isfinite(grids['a']) * (np.sqrt(grids['a']) > scale*0.75)
            ux, uy, vx, vy = [grids[i][gpi] for i in vel_grad_names]

            e1, e2, e3 = get_deformation(ux, uy, vx, vy)

            result[scale]['e1'] = e1
            result[scale]['e2'] = e2
            result[scale]['e3'] = e3
            result[scale]['a'] = grids['a'][gpi]
        result[scale0]['e1'] = p.e1
        result[scale0]['e2'] = p.e2
        result[scale0]['e3'] = p.e3
        result[scale0]['a'] = p.a
        return result

    def proc_one_date(self, date):
        pdefor = self.pf.filter(date)
        if len(pdefor) == 0:
            return None
        merged = self.merge_pairs(pdefor)
        c = self.coarse_grain(merged)
        m = compute_moments(c)
        return m

def dataframe_from_csv(ifile):
    """ Read RGPS data from a CSV file and convert to pandas DataFrame with OBS_DATE field """
    df = pd.read_csv(ifile)
    # ADD OBS_DATE
    for obs_year in df.OBS_YEAR.unique():
        df.loc[df.OBS_YEAR == obs_year, ('OBS_DATE')] = (pd.to_datetime(f'{obs_year}-1-1') + pd.to_timedelta(df[df.OBS_YEAR == obs_year].OBS_TIME, unit='D')).round('1s')

    # ADD INT INDEX
    u_img, u_idx, u_inv  = np.unique(df.IMAGE_ID, return_index=True, return_inverse=True)
    df.IMAGE_ID = np.arange(u_img.size)[u_inv]

    # MAKE A NEW DF AND SAVE
    df = df[['IMAGE_ID', 'GPID', 'OBS_DATE', 'X_MAP', 'Y_MAP', 'Q_FLAG']].copy().rename(columns={'IMAGE_ID':'i', 'GPID': 'g', 'OBS_DATE':'d', 'X_MAP':'x', 'Y_MAP':'y', 'Q_FLAG':'q'})
    return df

def get_rgps_pairs(df, date0, date1, min_time_diff=0.5, max_time_diff=5, min_size=10, r_min=0.12, a_max=200e6, cores=10):
    """ Find pairs of images in RGPS data and create a Pair objects """
    get_rgps_pairs_im2.df = df
    get_rgps_pairs_im2.max_time_diff = max_time_diff
    get_rgps_pairs_im2.min_time_diff = min_time_diff
    get_rgps_pairs_im2.min_size = min_size
    get_rgps_pairs_im2.r_min = r_min
    get_rgps_pairs_im2.a_max = a_max

    dfd = df[(df.d >= date0) * (df.d <= date1)]
    im2_idx = np.unique(dfd.i)
    if cores <= 1:
        pairs = list(map(get_rgps_pairs_im2, im2_idx))
    else:
        with Pool(cores) as p:
            pairs = p.map(get_rgps_pairs_im2, im2_idx)
    pairs = [j for i in pairs for j in i]
    return pairs

def get_rgps_pairs_im2(im2):
    df = get_rgps_pairs_im2.df
    max_time_diff = get_rgps_pairs_im2.max_time_diff
    min_time_diff = get_rgps_pairs_im2.min_time_diff
    min_size = get_rgps_pairs_im2.min_size
    r_min = get_rgps_pairs_im2.r_min
    a_max = get_rgps_pairs_im2.a_max

    pairs = []
    df2 = df[df.i == im2]
    d2 = df2.iloc[0].d

    gpi = (
        (df.d >= (d2 - timedelta(max_time_diff))) *
        (df.d <= (d2 - timedelta(min_time_diff)))
    )
    im1_idx = np.unique(df[gpi].i)
    for im1 in im1_idx:
        df1 = df[df.i == im1]
        int_ids, i1, i2 = np.intersect1d(df1.g, df2.g, return_indices=True)
        if i1.size > min_size:

            x0 = df1.iloc[i1].x.to_numpy()
            x1 = df2.iloc[i2].x.to_numpy()
            y0 = df1.iloc[i1].y.to_numpy()
            y1 = df2.iloc[i2].y.to_numpy()

            t, a, p = get_triangulation(x0, y0)
            r = np.sqrt(a) / p
            g = (r >= r_min) * (a <= a_max)
            if g[g].size == 0:
                continue

            pairs.append(Pair(
                x0 = x0.astype(np.float32),
                x1 = x1.astype(np.float32),
                y0 = y0.astype(np.float32),
                y1 = y1.astype(np.float32),
                d0 = df1.iloc[i1[0]].d,
                d1 = df2.iloc[i2[0]].d,
                t = t,
                a = a.astype(np.float32),
                p = p.astype(np.float32),
                g = g,
            ))
    return pairs

def measure(xt, yt):
    # side lengths (X,Y,tot)
    tri_x = np.diff(np.vstack([xt, xt[0]]), axis=0)
    tri_y = np.diff(np.vstack([yt, yt[0]]), axis=0)
    tri_s = np.hypot(tri_x, tri_y)
    # perimeter
    tri_p = np.sum(tri_s, axis=0)
    s = tri_p/2
    # area
    tri_a = np.sqrt(s * (s - tri_s[0]) * (s - tri_s[1]) * (s - tri_s[2]))
    return tri_a, tri_p

def get_triangulation(x, y):
    """ Triangulate input points and return trinagulation, area and perimeter """
    # get triangule indeces, area and perimeter
    tri = Triangulation(x, y)
    # coordinates of corners of each element
    xt, yt = [i[tri.triangles].T for i in (x, y)]
    # area, perimeter
    tri_a, tri_p = measure(xt, yt)
    return tri.triangles, tri_a, tri_p

def get_velocity_gradient_elems(x, y, u, a):
    """ Compute velocity gradient on input elements """
    # contour integrals of u and v [m/sec * m ==> m2/sec]
    ux = uy = 0
    for i0, i1 in zip([1, 2, 0], [0, 1, 2]):
        ux += (u[i0] + u[i1]) * (y[i0] - y[i1])
        uy -= (u[i0] + u[i1]) * (x[i0] - x[i1])
    # divide integral by double area [m2/sec / m2 ==> 1/sec]
    ux, uy =  [i / (2 * a) for i in (ux, uy)]
    return ux, uy

def merge_pairs(r_pairs, mfl, r_min, a_max, distance_upper_bound1, distance_upper_bound2, cores=10):
    merge_one_pair.mfl = mfl
    merge_one_pair.r_min = r_min
    merge_one_pair.a_max = a_max
    merge_one_pair.distance_upper_bound1 = distance_upper_bound1
    merge_one_pair.distance_upper_bound2 = distance_upper_bound2

    with Pool(10) as p:
        n_pairs = p.map(merge_one_pair, r_pairs)

    return n_pairs

def merge_one_pair(r):
    mfl = merge_one_pair.mfl
    r_min = merge_one_pair.r_min
    a_max = merge_one_pair.a_max
    distance_upper_bound1 = merge_one_pair.distance_upper_bound1
    distance_upper_bound2 = merge_one_pair.distance_upper_bound2

    _, nd0 = mfl.find_nearest(r.d0)
    _, nd1 = mfl.find_nearest(r.d1)

    if nd0 == nd1:
        #print('nd0 == nd1')
        return None

    xe0r = r.x0[r.t[r.g]].mean(axis=1)
    ye0r = r.y0[r.t[r.g]].mean(axis=1)
    rtree = KDTree(np.vstack([xe0r, ye0r]).T)

    x0n, y0n, ids0 = mfl.get_data(nd0)
    gpi = np.where(np.isfinite(x0n))[0]
    x0n, y0n, ids0 = [i[gpi] for i in [x0n, y0n, ids0]]

    x1n, y1n, ids1 = mfl.get_data(nd1)
    gpi = np.where(np.isfinite(x1n))[0]
    x1n, y1n, ids1 = [i[gpi] for i in [x1n, y1n, ids1]]

    # coordinates of nodes of common elements
    _, ids0i, ids1i = np.intersect1d(ids0, ids1, return_indices=True)
    x0n = x0n[ids0i]
    y0n = y0n[ids0i]
    x1n = x1n[ids1i]
    y1n = y1n[ids1i]

    if x0n.size < 3:
        #print('No common nodes in nextsim')
        return None

    dist, idx = rtree.query(np.vstack([x0n, y0n]).T, distance_upper_bound=distance_upper_bound1)
    idx = np.where(np.isfinite(dist))[0]
    if idx.size < 3:
        #print('No nextsim nodes close to rgps nodes')
        return None

    x0n = x0n[idx]
    y0n = y0n[idx]
    x1n = x1n[idx]
    y1n = y1n[idx]

    t0n, a0n, p0n = get_triangulation(x0n, y0n)
    r0n = np.sqrt(a0n) / p0n
    g0n = (r0n >= r_min) * (a0n <= a_max)

    xe0n = x0n[t0n].mean(axis=1)
    ye0n = y0n[t0n].mean(axis=1)

    dist, idx = rtree.query(np.vstack([xe0n, ye0n]).T, distance_upper_bound=distance_upper_bound2)
    idx = np.where(np.isfinite(dist))[0]

    if idx.size == 0:
        #print('No nextsim elements close to rgps nodes')
        return None

    t0n = t0n[idx]
    _, x0n, y0n = cleanup_triangulation(t0n, x0n, y0n)
    t0n, x1n, y1n = cleanup_triangulation(t0n, x1n, y1n)

    n = Pair(
        x0 = x0n.astype(np.float32),
        x1 = x1n.astype(np.float32),
        y0 = y0n.astype(np.float32),
        y1 = y1n.astype(np.float32),
        d0 = pd.Timestamp(nd0),
        d1 = pd.Timestamp(nd1),
        t = t0n,
        a = a0n[idx].astype(np.float32),
        p = p0n[idx].astype(np.float32),
        g = g0n[idx],
    )

    return n

def cleanup_triangulation(t, x, y):
    """ remove nodes which are not in the triangulation (node index not in t) """
    # find all node indeces and inverse index to reconstruct t
    i, i_inv = np.unique(t, return_inverse=True)
    # keep only unique node coordinates
    x = np.array(x[i])
    y = np.array(y[i])
    # recreate triangulation with unique nodes only
    t = np.arange(0, x.size)[i_inv].reshape((-1, 3))
    return t, x, y


def get_deformation_from_pair(p):
    if p is None:
        return None, None, None, None
    u, v = p.get_speed()
    ux, uy, vx, vy = p.get_velocity_gradients(u, v)
    e1, e2, e3 = get_deformation(ux, uy, vx, vy)
    return Defor(
        e1=e1.astype(np.float32),
        e2=e2.astype(np.float32),
        e3=e3.astype(np.float32),
        ux=ux.astype(np.float32),
        uy=uy.astype(np.float32),
        vx=vx.astype(np.float32),
        vy=vy.astype(np.float32),
    )

def get_deformation(ux, uy, vx, vy):
    # deformation components
    e1 = ux + vy
    e2 = np.hypot((ux - vy), (uy + vx))
    e3 = vx - uy
    return e1, e2, e3


def defor_to_aniso_con(p_d):
    """ Compute aniso only for connected elements """
    p,d = p_d
    min_e = defor_to_aniso_con.min_e
    power = defor_to_aniso_con.power
    edges_vec = defor_to_aniso_con.edges_vec
    min_size = defor_to_aniso_con.min_size

    if p is None:
        return None

    if p.x0.size < min_size:
        return None

    g = p.g
    t = p.t[g]
    x = p.x0
    y = p.y0
    e = np.hypot(d.e1[g], d.e2[g]) * DAY_SECONDS

    e_hi_idx = np.where(e > min_e)[0]
    t_hi = t[e_hi_idx]
    e_hi = e[e_hi_idx]

    tn = TriNeighbours(t_hi)
    aniso = defaultdict(list)
    for edges in edges_vec:
        # i - counter in t
        # i_hi - counter in t_hi
        for i_hi, i in enumerate(e_hi_idx):
            jj = np.array([i_hi] + tn.get_neighbours(i_hi, edges))
            if jj.size >= 3:#(2 + edges):
                xjj = x[t_hi[jj]].mean(axis=1)
                yjj = y[t_hi[jj]].mean(axis=1)
                ejj = e_hi[jj]
                anis_val, size_val = get_aniso_xye(xjj, yjj, ejj, power=power)
                aniso[f'ani|{edges}'].append(anis_val)
                aniso[f'siz|{edges}'].append(size_val)
                aniso[f'cnt|{edges}'].append(ejj.size)
                aniso[f'idx|{edges}'].append(i)

    for key in aniso:
        aniso[key] = np.array(aniso[key])

    return aniso


class TriNeighbours:
    neighbours = None
    def __init__(self, t):
        elem2edge, edge2elem = self.get_edge_elem_relationship(t)
        self.neighbours = []
        for i in range(t.shape[0]):
            neighbours_lists = [edge2elem[edge] for edge in elem2edge[i] if edge in edge2elem]
            neighbours_i = []
            for n1 in neighbours_lists:
                if len(n1) != 1:
                    for n2 in n1:
                        if n2 != i:
                            neighbours_i.append(n2)
            self.neighbours.append(neighbours_i)
        self.nneighbours = [len(n) for n in self.neighbours]

    def get_edge_elem_relationship(self, t):
        elem2edge = []
        edge2elem = defaultdict(list)
        for i, elem in enumerate(t):
            jj = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
            edges = [tuple(sorted(j)) for j in jj]
            elem2edge.append(edges)
            for edge in edges:
                edge2elem[edge].append(i)
        return elem2edge, edge2elem

    def get_neighbours(self, i, n=1, e=()):
        """ Get neighbours of element <i> crossing <n> edges

        Parameters
        ----------
        i : int, index of element
        n : int, number of edges to cross
        e : (int,), indeces to exclude

        Returns
        -------
        l : list, List of unique inidices of existing neighbor elements

        """
        # return list of existing immediate neigbours
        if n == 1:
            return self.neighbours[i]
        # recursively return list of neighbours after 1 edge crossing
        n2 = []
        for j in self.neighbours[i]:
            if j not in e:
                n2.extend(self.get_neighbours(j, n-1, e+(i,)))
        return list(set(self.neighbours[i] + n2))

    def get_neighbours_many(self, indices, n=1):
        neighbours_many = [self.get_neighbours(i, n=n) for i in indices]
        return np.unique(np.hstack(neighbours_many)).astype(int)

    def get_distance_to_border(self):
        dist = np.zeros(len(self.neighbours)) + np.nan
        border = np.where(np.array(self.nneighbours) < 3)[0]
        dist[border] = 0

        d = 1
        while np.any(np.isnan(dist)):
            for i in np.where(dist == d - 1)[0]:
                neibs = self.get_neighbours(i)
                for j in neibs:
                    if np.isnan(dist[j]):
                        dist[j] = d
            d += 1
        return dist


def get_aniso_xye(x, y, e, power=1):
    t = get_inertia_tensor_xye(x, y, e)
    eig_vals, eig_vecs = np.linalg.eig(t)
    eig_vals = np.abs(np.round(sorted(eig_vals), 4))
    if eig_vals[1] == 0:
        return 0., 0.
    return 1 - (eig_vals[0] / eig_vals[1])**power, eig_vals[1]

def get_inertia_tensor_xye(x, y, e):
    x1 = x - x.mean()
    y1 = y - y.mean()
    ixx = np.sum(e*x1**2)
    iyy = np.sum(e*y1**2)
    ixy = -np.sum(e*x1*y1)
    return np.array([[ixx, ixy], [ixy, iyy]])/e.sum()


def get_glcmd(p_d):
    p,d = p_d
    if p is None:
        return None
    l = get_glcmd.l
    e_min = get_glcmd.e_min
    e_max = get_glcmd.e_max
    d_vec= get_glcmd.d_vec

    g = p.g
    t = p.t
    e = np.hypot(d.e1, d.e2) * DAY_SECONDS

    glcmd = []
    dist = TriNeighbours(t).get_distance_to_border()
    for d in d_vec:
        tn, e_g = get_tn_eg(dist, t, g, e, d, l, e_min, e_max)
        glcmd.append(get_glcm(tn, e_g, d, l))
    glcmd = np.dstack(glcmd).reshape(l+1, l+1, len(d_vec), 1)
    return glcmd

def get_tn_eg(dist, t, g, e, d, l, e_min, e_max):
    # select good elements
    gpi = np.where(g * (dist > d))[0]
    t1 = t[gpi]
    e1 = e[gpi]
    tn1 = TriNeighbours(t1)

    # compute gray levels
    ee = np.clip(e1, 10**e_min, 10**e_max)
    e_g = np.floor(l * (np.log10(ee) - e_min) / (e_max - e_min)).astype(int)
    return tn1, e_g

def get_glcm(tn, e_g, d, l):
    glcm = np.zeros((l+1, l+1))
    for i in range(len(e_g)):
        neibs = tn.get_neighbours(i, d)
        if len(neibs) == 0:
            continue
        if d > 1:
            closer_neibs = tn.get_neighbours(i, d-1)
            neibs = list(set(neibs) - set(closer_neibs))
        if len(neibs) == 0:
            continue
        x = np.ones_like(neibs) * i
        glcm[e_g[neibs], e_g[x]] += 1
    return glcm

def compute_moments(etot, moment_powers=(1, 2, 3), factor=DAY_SECONDS, max_missing_values=2):
    scale_names = np.array(sorted(etot.keys()))
    scales = np.array([etot[scale_name]['a'].mean() for scale_name in scale_names])
    gpi = np.where(np.isfinite(scales))[0]
    if scales.size - gpi.size > max_missing_values:
        return None
    scales = scales[gpi]
    scale_names = scale_names[gpi]

    moms = []
    for scale_name in scale_names:
        e = np.hypot(etot[scale_name]['e1'], etot[scale_name]['e2']) * factor
        mom = [np.mean(e ** n) for n in moment_powers]
        moms.append(mom)
    moms = np.array(moms).T
    coefs = []
    moms2 = []
    for m in moms:
        coef = curve_fit(qval_on_l, np.log10(scales[gpi]), np.log10(m[gpi]))[0]

        moms2.append(10**qval_on_l(np.log10(scales[gpi]), *coef))
        coefs.append(coef)

    return dict(
        m = moms,
        c = np.array(coefs),
        m2 = np.array(moms2),
        s = np.array(scales)
    )

def qval_on_l(x, a, b):
    return -a*x + b


