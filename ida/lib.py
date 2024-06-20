from dataclasses import dataclass
from datetime import timedelta, datetime
import glob
from multiprocessing import Pool
import os

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pynextsim.nextsim_mesh import NextsimMesh
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


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

def get_triangulation(x, y):
    """ Triangulate input points and return trinagulation, area and perimeter """
    # get triangule indeces, area and perimeter
    tri = Triangulation(x, y)

    # coordinates of corners of each element
    xt, yt = [i[tri.triangles].T for i in (x, y)]

    # side lengths (X,Y,tot)
    tri_x = np.diff(np.vstack([xt, xt[0]]), axis=0)
    tri_y = np.diff(np.vstack([yt, yt[0]]), axis=0)
    tri_s = np.hypot(tri_x, tri_y)
    # perimeter
    tri_p = np.sum(tri_s, axis=0)
    s = tri_p/2
    # area
    tri_a = np.sqrt(s * (s - tri_s[0]) * (s - tri_s[1]) * (s - tri_s[2]))

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
