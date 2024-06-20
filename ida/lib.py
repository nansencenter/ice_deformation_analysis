from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
import pandas as pd


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
