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
from scipy.spatial import KDTree, cKDTree
from scipy.optimize import curve_fit

DAY_SECONDS = 24 * 60 * 60
DIST2COAST_NC = None


class BaseRunner:
    force = False
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def skip_processing(self, ofile):
        """ Skip processing if output file exists and should not be overwritten """
        if os.path.exists(ofile) and not self.force:
            return True
        return False


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
        try:
            m = NextsimMesh(ifile)
        except:
            print(f"Error reading {ifile}")
            raise

        try:
            x = m.nodes_x
        except:
            print(f"Error reading {ifile}")
            raise
        
        try:
            y = m.nodes_y
        except:
            print(f"Error reading {ifile}")
            raise
        
        try:
            i = m.get_var('id')
        except:
            print(f"Error reading {ifile}")
            raise

        self.data[idate] = dict(x = x, y = y, i = i,)

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

    def tripcolor(self, p, vmin=0, vmax=0.1, name='e1', units='d', cmap='plasma_r'):
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
    # minimum area/perimeter ratio to select for computing deformation
    # lower value allow sharper angles in triangles and increases number of elements
    min_ap_ratio = 0.18
    # minimum distance from coast
    min_dist=100
    # minimum displacement between nodes
    min_drift=1000
    # minimum size of triangulation
    # lower number increases number of triangles but increase chance of noise
    min_tri_size = 10
    # path to the dist2coas NPY file
    dist2coast_path = None
    
    def __init__(self, pairs, defor, resolution=10000, **kwargs):
        self.__dict__.update(kwargs)
        self.min_area = resolution ** 2 / 15
        self.max_area = resolution ** 2 * 15
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
    def __init__(self, pf, min_scale_factor=1.5, max_scale_factor=0.1, num_scales=10, skip_percentile=0.05):
        self.pf = pf
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.num_scales = num_scales
        self.skip_percentile = skip_percentile

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
        try:
            pair = MarsanPair(**m)
        except:
            import ipdb; ipdb.set_trace()
        return pair
        
    def compute_scales(self, merged):
        min_scale = np.median(merged.a)**0.5 * self.min_scale_factor
        max_scale = np.hypot(merged.x.max() - merged.x.min(), merged.y.max() - merged.y.min()) * self.max_scale_factor
        scales = np.round(np.logspace(np.log10(min_scale), np.log10(max_scale), num=self.num_scales)/100).astype(int)*100
        return scales

    def coarse_grain(self, merged, scales):
        # remove extreme outliers
        me_e4 = np.hypot(merged.e1, merged.e2)
        gpi = (me_e4 > np.percentile(me_e4, self.skip_percentile)) * (me_e4 < np.percentile(me_e4, 100-self.skip_percentile))
        for name in merged.__dict__:
            merged.__dict__[name] = merged.__dict__[name][gpi]

        etot = {}
        etot_grid = {}
        for scale in scales:
            merged_cols = ((merged.x - merged.x.min()) / scale).astype(int)
            merged_rows = ((merged.y.max() - merged.y) / scale).astype(int)

            grids = {}
            vel_grad_names = ['ux', 'uy', 'vx', 'vy']
            sum_names = ['a'] +  vel_grad_names
            for name in sum_names:
                sum_array = np.zeros((merged_rows.max() + 1, merged_cols.max() + 1))
                if name == 'a':
                    values_to_sum = merged.a
                else:
                    values_to_sum = merged.__dict__[name] * merged.a
                np.add.at(sum_array, (merged_rows, merged_cols), values_to_sum)
                grids[name] = sum_array

            grids['a'][grids['a'] == 0] = np.nan
            for name in vel_grad_names:    
                grids[name] /= grids['a']

            gpi = np.isfinite(grids['a'])
            ux, uy, vx, vy = [grids[i][gpi] for i in vel_grad_names]
            e1, e2, _ = get_deformation(ux, uy, vx, vy)
            etot[scale] = np.hypot(e1, e2) * DAY_SECONDS

            etot_grid[scale] = np.zeros_like(grids['a']) + np.nan
            etot_grid[scale][gpi] = etot[scale]
        return etot, etot_grid

    def proc_one_date(self, date):
        pdefor = self.pf.filter(date)
        if len(pdefor) == 0:
            return None
        merged = self.merge_pairs(pdefor)
        scales = self.compute_scales(merged)
        etot, _ = self.coarse_grain(merged, scales)
        m = compute_moments(etot)
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

def thin_poisson(x, y, r, seed=None):
    rng = np.random.default_rng(seed)
    pts = np.column_stack((x, y))
    n = len(pts)
    order = rng.permutation(n)  # randomized order reduces bias
    tree = cKDTree(pts)
    keep = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)

    for i in order:
        if not available[i]:
            continue
        keep[i] = True
        # Remove all points within r of the selected point
        neigh = tree.query_ball_point(pts[i], r)
        available[neigh] = False
        available[i] = True  # keep remains selected
    return keep


class GetSatPairs:
    min_time_diff = 0.5
    max_time_diff = 7
    min_size = 100
    r_min = 0.12
    a_max = 200e6
    d_min = 200
    poisson_disk_radius = None
    exclude_regions = None
    cores = 4

    def __init__(self, df, **kwargs):
        self.df = df
        self.__dict__.update(kwargs)

    def get_all_pairs(self, date0, date1):
        """ Find pairs of images in RGPS or S1 data and create a Pair objects """
        
        dfd = self.df[(self.df.d >= date0) * (self.df.d <= date1)]
        im2_idx = np.unique(dfd.i)
        if self.cores <= 1:
            pairs = list(map(self.get_pairs_for_img2, im2_idx))
        else:
            with Pool(self.cores) as p:
                pairs = p.map(self.get_pairs_for_img2, im2_idx)
        pairs = [j for i in pairs for j in i]
        return pairs

    def get_pairs_for_img2(self, im2):
        pairs = []
        df2 = self.df[self.df.i == im2]
        d2 = df2.iloc[0].d

        gpi = (
            (self.df.d >= (d2 - timedelta(self.max_time_diff))) *
            (self.df.d <= (d2 - timedelta(self.min_time_diff)))
        )
        im1_idx = np.unique(self.df[gpi].i)
        for im1 in im1_idx:
            df1 = self.df[self.df.i == im1]
            int_ids, i1, i2 = np.intersect1d(df1.g, df2.g, return_indices=True)
            if i1.size < self.min_size:
                continue
            x0 = df1.iloc[i1].x.to_numpy()
            x1 = df2.iloc[i2].x.to_numpy()
            y0 = df1.iloc[i1].y.to_numpy()
            y1 = df2.iloc[i2].y.to_numpy()

            if self.exclude_regions is not None:
                mask_exclude = np.zeros(x0.shape, dtype=bool)
                for exclude_region in self.exclude_regions:
                    mask_exclude += ((x0 > exclude_region[0]) *
                                     (x0 < exclude_region[1]) *
                                     (y0 > exclude_region[2]) * 
                                     (y0 < exclude_region[3]))
                x0 = x0[~mask_exclude]
                y0 = y0[~mask_exclude]
                x1 = x1[~mask_exclude]
                y1 = y1[~mask_exclude]
                if x0.size < self.min_size:
                    continue

            # filter individual points by minimum drift
            drift = np.hypot(x1 - x0, y1 - y0)
            mask = drift >= self.d_min
            x0 = x0[mask]
            x1 = x1[mask]
            y0 = y0[mask]
            y1 = y1[mask]
            if x0.size < self.min_size:
                continue
            
            # ADD PAIR FILTERING
            if self.poisson_disk_radius is not None:
                mask = thin_poisson(x0, y0, r=self.poisson_disk_radius)
                x0 = x0[mask]
                x1 = x1[mask]
                y0 = y0[mask]
                y1 = y1[mask]
                if x0.size < self.min_size:
                    continue

            t, a, p, r = get_triangulation(x0, y0)
            g = (r >= self.r_min) * (a <= self.a_max)
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

def jacobian(x0, y0, x1, y1, x2, y2):
    return (x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)

def get_area(xt, yt):
    return .5*jacobian(xt[0], yt[0], xt[1], yt[1], xt[2], yt[2])

def measure(xt, yt):
    dx = np.diff(np.vstack([xt, xt[0]]), axis=0)
    dy = np.diff(np.vstack([yt, yt[0]]), axis=0)
    edges = np.hypot(dx, dy)
    perim = edges.sum(axis=0)
    area = get_area(xt, yt)
    ap_ratio = area**0.5 / perim
    return area, perim, ap_ratio

def get_triangulation(x, y):
    """ Triangulate input points and return trinagulation, area and perimeter """
    # get triangule indeces, area and perimeter
    tri = Triangulation(x, y)
    # coordinates of corners of each element
    xt, yt = x[tri.triangles].T, y[tri.triangles].T
    # area, perimeter
    tri_a, tri_p, tri_r = measure(xt, yt)

    return tri.triangles, tri_a, tri_p, tri_r

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

def sample_nextsim_pairs(r_pairs, mfl, r_min, a_max, distance_upper_bound1, distance_upper_bound2, cores):
    """ NB: Outdated function. It was used in script c_merge_nextsim_rgps to sample nextsim outputs around RGPS obs
        collocate_pairs is used instead now
    """
    sample_one_pair = SampleOnePair(mfl, r_min, a_max, distance_upper_bound1, distance_upper_bound2)
    if cores <= 1:
        n_pairs = list(map(sample_one_pair, r_pairs))
    with Pool(cores) as p:
        n_pairs = p.map(sample_one_pair, r_pairs)

    return n_pairs

class SampleOnePair:
    """ NB: Outdated class. It was used in script c_merge_nextsim_rgps to sample nextsim outputs around RGPS obs
        MergeSatNextPair is used instead now.
    """
    def __init__(self, mfl, r_min, a_max, distance_upper_bound1, distance_upper_bound2):
        self.mfl = mfl
        self.r_min = r_min
        self.a_max = a_max
        self.distance_upper_bound1 = distance_upper_bound1
        self.distance_upper_bound2 = distance_upper_bound2

    def __call__(self, r):
        _, nd0 = self.mfl.find_nearest(r.d0)
        _, nd1 = self.mfl.find_nearest(r.d1)

        if nd0 == nd1:
            #print('nd0 == nd1')
            return None

        xe0r = r.x0[r.t[r.g]].mean(axis=1)
        ye0r = r.y0[r.t[r.g]].mean(axis=1)
        rtree = KDTree(np.vstack([xe0r, ye0r]).T)

        x0n, y0n, ids0 = self.mfl.get_data(nd0)
        gpi = np.where(np.isfinite(x0n))[0]
        x0n, y0n, ids0 = [i[gpi] for i in [x0n, y0n, ids0]]

        x1n, y1n, ids1 = self.mfl.get_data(nd1)
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

        dist, idx = rtree.query(np.vstack([x0n, y0n]).T, distance_upper_bound=self.distance_upper_bound1)
        idx = np.where(np.isfinite(dist))[0]
        if idx.size < 3:
            #print('No nextsim nodes close to rgps nodes')
            return None

        x0n = x0n[idx]
        y0n = y0n[idx]
        x1n = x1n[idx]
        y1n = y1n[idx]

        t0n, a0n, p0n, r0n = get_triangulation(x0n, y0n)
        g0n = (r0n >= self.r_min) * (a0n <= self.a_max)

        xe0n = x0n[t0n].mean(axis=1)
        ye0n = y0n[t0n].mean(axis=1)

        dist, idx = rtree.query(np.vstack([xe0n, ye0n]).T, distance_upper_bound=self.distance_upper_bound2)
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


class DeformationToAnisotropyConnected:
    def __init__(self, min_e=0.1, power=2, edges_vec=None, min_size=3, min_neibors=3):
        self.min_e = min_e
        self.power = power
        self.edges_vec = edges_vec if edges_vec is not None else [1, 2, 3]
        self.min_size = min_size
        self.min_neibors = min_neibors

    def __call__(self, p_d):
        """ Compute aniso only for connected elements """
        p,d = p_d
        min_e = self.min_e
        power = self.power
        edges_vec = self.edges_vec
        min_size = self.min_size

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
                if jj.size >= self.min_neibors:
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


class GetGLCM:
    def __init__(self, e_min, e_max, l, d_vec):
        self.e_min = e_min
        self.e_max = e_max
        self.l = l
        self.d_vec = d_vec

    def __call__(self, p_d):
        p,d = p_d
        if p is None:
            return None
        l = self.l
        e_min = self.e_min
        e_max = self.e_max
        d_vec = self.d_vec

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

def compute_moments(etot, moment_powers=(1, 2, 3), exclude_max_error_scale=True):
    scales = np.array(sorted(etot.keys()))

    moms = []
    for scale in scales:
        e = etot[scale]
        mom = [np.mean(e ** n) for n in moment_powers]
        moms.append(mom)
    
    moms = np.array(moms).T
    coefs = []
    moms2 = []
    for m in moms:
        coef = curve_fit(qval_on_l, np.log10(scales), np.log10(m))[0]
        m_eval = 10**qval_on_l(np.log10(scales), *coef)
        abs_err = np.abs((m - m_eval))
        
        # exclude max  error on some scale
        if abs_err.max() > np.std(abs_err)*3 and exclude_max_error_scale:
            m_ = m[abs_err < abs_err.max()]
            scales_ = scales[abs_err < abs_err.max()]
            coef = curve_fit(qval_on_l, np.log10(scales_), np.log10(m_))[0]
            m_eval = 10**qval_on_l(np.log10(scales), *coef)

        moms2.append(m_eval)
        coefs.append(coef)
    
    return dict(
        m = moms,
        c = np.array(coefs),
        m2 = np.array(moms2),
        s = np.array(scales)
    )

def qval_on_l(x, a, b):
    return -a*x + b

def pair_from_nextsim_snapshots(f0, f1, d0, d1, r_min=0.12, a_max=200e6):
    m0 = NextsimMesh(f0)
    m1 = NextsimMesh(f1)
    x0 = m0.nodes_x
    y0 = m0.nodes_y
    i0 = m0.get_var('id')

    x1 = m1.nodes_x
    y1 = m1.nodes_y
    i1 = m1.get_var('id')

    # indices of nodes common to 0 and 1
    _, ids0i, ids1i = np.intersect1d(i0, i1, return_indices=True)

    # coordinates of nodes of common elements
    x0n = x0[ids0i]
    y0n = y0[ids0i]
    x1n = x1[ids1i]
    y1n = y1[ids1i]
    t, a, p, r = get_triangulation(x0n, y0n)
    g = (r >= r_min) * (a <= a_max)

    t0 = Triangulation(m0.nodes_x, m0.nodes_y, m0.indices)
    try:
        fi0 = t0.get_trifinder()
    except:
        print('Cannot get trifinder of m0 triangulation')
    else:
        f0i = fi0(x0n[t].mean(axis=1), y0n[t].mean(axis=1))
        g[f0i == -1] = False

    return Pair(x0n, x1n, y0n, y1n, d0, d1, t, a, p, g)

def get_velocity_gradient_nodes(x, y, u, v):
    """ Compute velocity gradient on input nodes """
    # get triangule indeces, area and perimeter
    tri_i, tri_a, tri_p, tri_r = get_triangulation(x, y)

    # coordinates and speeds of corners of each element
    xt, yt, ut, vt = [i[tri_i].T for i in (x, y, u, v)]

    #ux, uy, vx, vy = velocity_integrals(xt, yt, ut, vt, tri_a)
    ux, uy = get_velocity_gradient_elems(xt, yt, ut, tri_a)
    vx, vy = get_velocity_gradient_elems(xt, yt, vt, tri_a)

    return ux, uy, vx, vy, tri_a, tri_p, tri_i


def filter_e1(e1, e2, tria, n, min_et, use_median=False):
    """ Filter divergence values based on the magnitude of total deformation in the neighbours.

    Parameters
    ----------
    e1 : np.ndarray
        Divergence values.
    e2 : np.ndarray
        Shear values.
    tria : np.ndarray
        Triangulation array.
    n : int
        Number of neighbours to consider.
    min_et : float
        Minimum total deformation value to consider.
    use_median : bool
        Use median instead of mean.

    Returns
    -------
    np.ndarray
        Filtered divergence values.

    """
    # get neighbourhood
    tn = TriNeighbours(tria)
    e1f = np.array(e1)
    e_tot = np.hypot(e1, e2)
     # loop over all elements
    for i, eti in enumerate(e_tot):
        if eti < min_et:
            continue
        # find neighbours with total deformation above threshold
        j0 = np.array(tn.get_neighbours(i, n=n))
        if j0.size < 1:
            continue
        etj0 = e_tot[j0]
        j1 = j0[etj0 > min_et]

        if j1.size < 1:
            continue
        # average divergence of neighbours using median or signed mean of absolute values
        if use_median:
            e1f[i] = np.nanmedian(e1[j1])
        else:
            e1f[i] = np.nanmean(e1[j1]**2)**0.5 * np.sign(np.median(e1[j1]))
    return e1f

def find_mututual_nearest_neighbours(points_set1, points_set2):
    # Mutual nearest neighbors using cKDTree: select points in set1 that are nearest to set2
    # and whose matched set2 point has them as its nearest in set1.

    # Build trees
    tree1 = cKDTree(points_set1)
    tree2 = cKDTree(points_set2)

    # Nearest neighbor in the other set
    d12, nn2_of_1 = tree2.query(points_set1)  # for each set1 point -> nearest set2 index
    d21, nn1_of_2 = tree1.query(points_set2)  # for each set2 point -> nearest set1 index

    # Indices of mutual nearest neighbors
    mutual_points2 = np.nonzero(nn2_of_1[nn1_of_2] == np.arange(len(nn1_of_2)))[0]
    mutual_points1 = nn1_of_2[mutual_points2]
    return mutual_points1, mutual_points2

def get_subset_pair(idx, x0, y0, x1, y1, d0, d1, r_min, a_max):
    t0, a0, p0, r0 = get_triangulation(x0[idx], y0[idx])
    g0 = (r0 >= r_min) * (a0 <= a_max)

    pair = Pair(
        x0 = x0[idx].astype(np.float32),
        x1 = x1[idx].astype(np.float32),
        y0 = y0[idx].astype(np.float32),
        y1 = y1[idx].astype(np.float32),
        d0 = pd.Timestamp(d0),
        d1 = pd.Timestamp(d1),
        t = t0,
        a = a0.astype(np.float32),
        p = p0.astype(np.float32),
        g = g0,
    )

    return pair

class CollocatePair:
    def __init__(self, mfl, r_min, a_max, distance_upper_bound1):
        self.mfl = mfl
        self.r_min = r_min
        self.a_max = a_max
        self.distance_upper_bound1 = distance_upper_bound1

    def __call__(self, r):
        _, nd0 = self.mfl.find_nearest(r.d0)
        _, nd1 = self.mfl.find_nearest(r.d1)

        if nd0 == nd1:
            #print('nd0 == nd1')
            return None

        xe0r = r.x0[r.t[r.g]].mean(axis=1)
        ye0r = r.y0[r.t[r.g]].mean(axis=1)
        rtree = cKDTree(np.vstack([xe0r, ye0r]).T)

        x0n, y0n, ids0 = self.mfl.get_data(nd0)
        gpi = np.where(np.isfinite(x0n))[0]
        x0n, y0n, ids0 = [i[gpi] for i in [x0n, y0n, ids0]]

        x1n, y1n, ids1 = self.mfl.get_data(nd1)
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

        dist, idx = rtree.query(np.vstack([x0n, y0n]).T, distance_upper_bound=self.distance_upper_bound1)
        idx = np.where(np.isfinite(dist))[0]
        if idx.size < 3:
            #print('No nextsim nodes close to rgps nodes')
            return None

        x0n = x0n[idx]
        y0n = y0n[idx]
        x1n = x1n[idx]
        y1n = y1n[idx]

        mpi1, mpi2 = find_mututual_nearest_neighbours(np.column_stack([r.x0, r.y0]), np.column_stack([x0n, y0n]))

        r_pair = get_subset_pair(mpi1, r.x0, r.y0, r.x1, r.y1, r.d0, r.d1, self.r_min, self.a_max)
        n_pair = get_subset_pair(mpi2, x0n, y0n, x1n, y1n, nd0, nd1, self.r_min, self.a_max)

        return r_pair, n_pair

def collocate_pairs(r_pairs, mfl, r_min, a_max, distance_upper_bound1, cores):
    collocate_pair = CollocatePair(mfl, r_min, a_max, distance_upper_bound1)
    if cores <= 1:
        n_pairs = list(map(collocate_pair, r_pairs))
    with Pool(cores) as p:
        n_pairs = p.map(collocate_pair, r_pairs)
    return n_pairs

def get_nn_distances(p):
    points = np.column_stack((p.x0, p.y0))
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    return distances[:, 1]
