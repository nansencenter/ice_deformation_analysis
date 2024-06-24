from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
import pandas as pd
from pynextsim.lib import jacobian
from scipy.stats import gaussian_kde
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import gaussian_filter

from pysida.lkf_detection import lkf_detect_eps

def fill_nan_gaps(defor, distance=5):
    """ Fill gaps in input raster with deformation (inplace)
    Parameters
    ----------
    defor : 2D numpy.array
        Ratser with deformation field
    distance : int
        Minimum size of gap to fill
    Returns
    -------
    defor : 2D numpy.array
        Raster with deformation field with gaps filled
    """
    dist, indi = distance_transform_edt(
        np.isnan(defor),
        return_distances=True,
        return_indices=True)
    gpi = dist <= distance
    r,c = indi[:,gpi]
    defor[gpi] = defor[r,c]
    return defor

def get_intersecting_segments(seg, max_dist=3, min_length=10):
    # find sufficient large segments and keep their indices
    seg_l = []
    seg_i = []
    for i, s in enumerate(seg):
        if s.shape[1] > min_length:
            seg_l.append(s)
            seg_i.append(i)
    # stack  sufficiently large segments
    segs = np.hstack(seg_l)
    # indices of stacked segments
    segi = np.hstack([np.ones(s.shape[1]) * i for (i, s) in enumerate(seg_l)])
    # 2D mask of not same segments
    notsamei = (segi - segi[None].T) != 0
    # distance between all points of stacked segments
    dist = np.hypot(segs[0] - segs[0][None].T, segs[1] - segs[1][None].T)
    # 2D mask of close enough points
    distoki = (dist > 0) * (dist < max_dist)
    # 2D mask of close enough points from different segments
    gpi = notsamei * distoki
    if np.all(~gpi):
        return [], []
    # indeces of valid points
    gpiw = np.where(gpi)
    # take only lower triangle
    gpil = np.tril(gpi)
    # indeces of valid points
    gpiw = np.where(gpil)
    pairi0 = gpiw[0]
    pairi1 = gpiw[1]
    # indeces of close segments
    seg_close_i = np.vstack([segi[pairi0], segi[pairi1]])
    # unique indeces of close segments
    seg_close_i  = np.unique(np.vstack([segi[pairi0], segi[pairi1]]), axis=1)
    # return indices of original segments
    return np.array(seg_i)[seg_close_i.astype(int)].astype(int)

def get_oriented_segs(segs, seg_orig_i0, seg_orig_i1, vort, min_length=3):
    """
    From inputs segments <segs> and indices of intersecting segments
    create two lists of split and orieneted segments.
    """
    all_segs0 = []
    all_segs1 = []
    all_vort0 = []
    all_vort1 = []
    for i0, i1 in zip(seg_orig_i0, seg_orig_i1):
        seg0 = segs[i0]
        seg1 = segs[i1]
        dist = np.hypot(seg0[0][None].T - seg1[0][None], seg0[1][None].T - seg1[1][None])
        idx0, idx1 = np.unravel_index(np.argmin(dist), dist.shape)

        segs0, vort0 = orient_split_seg(seg0, idx0, vort[i0])
        segs1, vort1 = orient_split_seg(seg1, idx1, vort[i1])
        for s0, v0 in zip(segs0, vort0):
            if s0.shape[1] < min_length:
                continue
            for s1, v1 in zip(segs1, vort1):
                if s1.shape[1] < min_length:
                    continue
                all_segs0.append(s0)
                all_segs1.append(s1)
                all_vort0.append(v0)
                all_vort1.append(v1)
    return all_segs0, all_segs1, all_vort0, all_vort1

def orient_split_seg(seg, idx, vort, min_seg_length=2):
    """
    Given index of intersection with other segment <idx> orient the output segment
    to start from idx. If idx is in the middle: return two oriented segments.
    """
    if idx == 0:
        segs = [seg]
        vrts = [vort]
    if idx == seg.shape[1] - 1:
        segs = [seg[:, ::-1]]
        vrts = [vort[::-1]]
    if 0 < idx < seg.shape[1] - 1:
        segs = [
            seg[:, :idx][:, ::-1],
            seg[:, idx:],
        ]
        vrts = [
            vort[:idx][::-1],
            vort[idx:],
        ]

    segs = [s for s in segs if s.shape[1] > min_seg_length]
    vrts = [v for v in vrts if len(v) > min_seg_length]
    return segs, vrts

def get_smooth_orientations(segs, to180, deg):
    oris = []
    for s in segs:
        s = invert_y(s)
        o = get_smooth_orientation(s, to180=to180, deg=deg)
        oris.append(o)
    return oris

def invert_y(seg):
    return np.vstack([seg[0], -seg[1]])

def get_smooth_orientation(s, to180=True, **kwargs):
    """ Get smoothed orientation for each point in a single segment """
    s1 = seg_approximated(s, **kwargs)
    dx = np.gradient(s1[0])
    dy = np.gradient(s1[1])
    angle = np.degrees(np.arctan2(dy, dx))
    if to180:
        angle[angle < 0] += 180
    return angle

def get_orientation(s, to180=True, **kwargs):
    """ Get average orientation of a single segment """
    dxp, dyp = seg_approximated(s, **kwargs)
    angle = np.degrees(np.arctan2(dyp[-1] - dyp[0], dxp[-1] - dxp[0]))
    if to180 and angle < 0:
        angle += 180
    return angle

def get_orientations(seg, **kwargs):
    """ Get orientation of all segment """
    return np.array([get_orientation(s, **kwargs) for s in seg])

def seg_approximated(s, deg=1, **kwargs):
    """
    Return approximated x, y as function of distance from starting point.
    """
    x0, y0, px, py = seg_polyfit(s, deg=deg, **kwargs)
    x, y = seg_polyval(x0, y0, px, py, s[0], s[1], deg=deg, **kwargs)
    return x, y

def seg_polyfit(s, deg=1, **kwargs):
    """
    Fit polynomial for approximating the dependency of x,y coordinates
    on distance from the starting point """
    x0, y0 = s[:, 0]
    dd = seg_distance(x0, y0, s[0], s[1], **kwargs)
    px = polyfit_float(dd, s[0]-s[0][0], deg=deg)
    py = polyfit_float(dd, s[1]-s[1][0], deg=deg)
    return x0, y0, px, py

def seg_distance(x0, y0, x, y, euclidean=True, **kwargs):
    """ Distance from starting point """
    if euclidean:
        # Eucledian distance of each point from starting point
        dx = x - x0
        dy = y - y0
        return np.hypot(dx, dy)

    # Path length from starting point
    s = np.array([np.hstack([x0, x]), np.hstack([y0, y])])
    return np.cumsum(np.hypot(*np.abs(np.diff(s))))

def polyfit_float(x, y, deg=1):
    """ Fit Y with polynom with order different from 2 (e.g. 1.5)"""
    if deg in [1, 2, 3]:
        return np.polyfit(x, y, deg=deg)
    a = np.vstack([np.ones(len(x)), x, x**deg]).T
    return np.linalg.lstsq(a, y, rcond=None)[0]

def seg_polyval(x0, y0, px, py, x, y, deg=1, **kwargs):
    """ Evaluate segment polynomial on input x,y point """
    dd = seg_distance(x0, y0, x, y, **kwargs)
    xp = x0 + polyval_float(px, dd, deg=deg)
    yp = y0 + polyval_float(py, dd, deg=deg)
    return np.vstack([xp, yp])

def polyval_float(p, x, deg=1):
    """ Eval Y with polynom with order different from 2 (e.g. 1.5)"""
    if deg in [1,2,3]:
        return np.polyval(p, x)
    return p[0] + p[1]*x + p[2]*x**deg

def acute_angle(d1, d2):
    a = (d2 - d1) % 360
    if a > 180:
        a -= 180
    if a > 90:
        a = 180 - a
    return a

def get_obtuse_angles(oris0, oris1, deg=1):
    angles = []
    for o0, o1 in zip(oris0, oris1):
        a = obtuse_angle(o0[0], o1[0])
        angles.append(a)
    return angles

def obtuse_angle(d0, d1):
    a1 = d0 - d1
    if a1 > 180:
        a1 = 360 - a1
    if a1 < -180:
        a1 = 360 + a1
    a1 = abs(a1)
    return a1

def get_dilation_angles(oris0, oris1, vort0, vort1, deg=1, min_vort_significance=1):
    angles = []
    for o0, o1, v0, v1 in zip(oris0, oris1, vort0, vort1):
        vm0 = np.mean(v0) / np.std(v0)
        vm1 = np.mean(v1) / np.std(v1)

        if np.abs(vm0) > min_vort_significance:
            vs0 = np.sign(vm0)
        else:
            vs0 = 0
        if np.abs(vm1) > min_vort_significance:
            vs1 = np.sign(vm1)
        else:
            vs1 = 0

        if vs0 > vs1:
            angles.append(dilation_angle(o1[0], o0[0]))
        elif vs1 > vs0:
            angles.append(dilation_angle(o0[0], o1[0]))
    return angles

def dilation_angle(d_pos, d_neg):
    a = (d_neg - d_pos) % 360
    if a > 180:
        a -= 180
    return a

class Rasterizer:
    x_lft=-2500000.0
    x_rht=300000
    y_top=2500000
    y_bot=-1000000.0
    fill_nan_gaps_dist=3
    gaus_filt_size=0.75
    gaus_filt_trunc=1
    daysec = 24 * 60 * 60
    res=12500
    res_step=2

    vlims = [
        [-0.015, 0.015],
        [0, 0.07],
        [-0.08, 0.08],
    ]

    def __init__(self):
        self.x_grd, self.y_grd = np.meshgrid(
            np.arange(self.x_lft, self.x_rht, self.res / self.res_step),
            np.arange(self.y_bot, self.y_top, self.res / self.res_step),
        )

    def rasterize(self, pdefor):
        eee = [
            np.zeros(self.x_grd.shape) + np.nan,
            np.zeros(self.x_grd.shape) + np.nan,
            np.zeros(self.x_grd.shape) + np.nan,
            np.zeros(self.x_grd.shape) + np.nan,
        ]
        for p in pdefor:
            gpi_r, gpi_c = np.where((
                (self.x_grd >= p.x1.min()) *
                (self.x_grd <= p.x1.max()) *
                (self.y_grd >= p.y1.min()) *
                (self.y_grd <= p.y1.max())
            ))
            tri = Triangulation(p.x1, p.y1, p.t)
            try:
                tfi = tri.get_trifinder()
            except:
                #tt = remove_nodes_w_negative_area(p.x1, p.y1, p.t)
                tt = remove_negative_elements(p.x1, p.y1, p.t)
                tri = Triangulation(p.x1, p.y1, tt)
                try:
                    tfi = tri.get_trifinder()
                except:
                    bad_elements = set(np.where(np.sum((tri.neighbors == -1).astype(int), axis=1) > 0)[0])
                    good_elements = np.array(list(set(range(tt.shape[0])) - bad_elements))
                    ttt = tt[good_elements]
                    tri = Triangulation(p.x1, p.y1, ttt)
                    tri.get_trifinder()

            i = tfi(self.x_grd[gpi_r, gpi_c], self.y_grd[gpi_r, gpi_c])
            if i.size == 0:
                continue
            gpi_r, gpi_c, i = [j[i >=0] for j in [gpi_r, gpi_c, i]]
            if i.size == 0:
                continue
            peeee = [p.e1, p.e2, p.e3, np.hypot(p.e1, p.e2)]
            for j, e in enumerate(peeee):
                tmp_grd = np.zeros(self.x_grd.shape) + np.nan
                tmp_grd[gpi_r, gpi_c] = e[i]
                eee[j][np.isfinite(tmp_grd)] = tmp_grd[np.isfinite(tmp_grd)]
        for j in range(4):
            eee[j] *= self.daysec
        return eee

    def subsample(self, eee):
        eee_sub = []
        for e in eee:
            egrdf0 = fill_nan_gaps(np.array(e), self.fill_nan_gaps_dist)
            egrdf1 = gaussian_filter(egrdf0, self.gaus_filt_size, truncate=self.gaus_filt_trunc)[::self.res_step, ::self.res_step]
            eee_sub.append(egrdf1)
        etot_sub = np.hypot(eee_sub[0], eee_sub[1])
        eee_sub.append(etot_sub)
        return eee_sub

    def __call__(self, *args):
        eee = self.rasterize(*args)
        eee_sub = self.subsample(eee)
        return eee_sub


class LKFDetector:
    lkf_detect_eps_params = dict(
        dog_thres = 15,
        dis_thres = 4,
        ellp_fac = 2,
        angle_thres = 35,
        eps_thres = 1.25,
        lmin = 3,
        min_kernel = 1,
        max_kernel = 5,
    )

    def __init__(self, pair_filter, rasterizer, **kwargs):
        self.lkf_detect_eps_params.update(kwargs)
        self.pair_filter = pair_filter
        self.rasterizer = rasterizer

    def detect(self, eee, e_i):
        lkfs = lkf_detect_eps(eee[e_i], **self.lkf_detect_eps_params)
        defs = [[e[l[0], l[1]] for l in lkfs] for e in [eee[0], eee[1], eee[2]]]
        return lkfs, defs

    def clean_nan_lkfs(self, lkfs, defs, min_lkf_size=3):
        lkfsr, lkfsc = zip(*lkfs)

        new_lkfsr, new_lkfsc, new_defs0,  new_defs1, new_defs2 = [], [], [], [], []
        for def0, def1, def2, lkfr, lkfc in zip(defs[0], defs[1], defs[2], lkfsr, lkfsc):
            gpi = np.where(np.isfinite(def0))[0]
            if gpi.size < min_lkf_size:
                continue
            else:
                new_lkfsr.append(lkfr[gpi])
                new_lkfsc.append(lkfc[gpi])
                new_defs0.append(def0[gpi])
                new_defs1.append(def1[gpi])
                new_defs2.append(def2[gpi])

        new_lkfs = [np.array([r, c]) for (r,c) in zip(new_lkfsr, new_lkfsc)]
        new_defs = [new_defs0,  new_defs1, new_defs2]
        return new_lkfs, new_defs

    def plot(self, e_plot, lkfs, figsize=(10,10)):
        _, axs = plt.subplots(1, 1, figsize=figsize)
        imsh = axs.imshow(e_plot, interpolation='nearest', cmap='gray')
        for r,c in lkfs:
            axs.plot(c, r)
        plt.colorbar(imsh, ax=axs)
        plt.show()

    def proc_one_date(self, date, e_i=3):
        pdefor = self.pair_filter.filter(date)
        eee = self.rasterizer.rasterize(pdefor)
        eee_sub = self.rasterizer.subsample(eee)
        try:
            lkfs, defs = self.detect(eee_sub, e_i)
        except ValueError:
            print('Error in LK detection on', date)
            return None, None
        lkfs, defs = self.clean_nan_lkfs(lkfs, defs)
        return lkfs, defs


def get_lkf_intersection_angs(lkfs, defs, deg=1, max_dist=3, min_length=5, min_vort_significance=1, **kwargs):
    # deg, degree for segment approximation
    # max_dist, maximum distance between LKFs to consider for intersection [pixels of 12.5 km]
    # min_length, minimum length of LKFs [pix]
    # min_vort_significance, minimum ratio of mean to std vorticity for dilation angle

    seg_orig_i0, seg_orig_i1 = get_intersecting_segments(lkfs, max_dist=max_dist, min_length=min_length)
    segs0, segs1, vort0, vort1 =  get_oriented_segs(lkfs, seg_orig_i0, seg_orig_i1, defs[2], min_length=min_length)
    oris0 = get_smooth_orientations(segs0, False, deg=deg)
    oris1 = get_smooth_orientations(segs1, False, deg=deg)
    dilat_angles = get_dilation_angles(oris0, oris1, vort0, vort1, min_vort_significance=min_vort_significance)
    obtus_angles = get_obtuse_angles(oris0, oris1)
    return dilat_angles, obtus_angles

def get_average_lkf_stats(lkfs_defs, dates, min_num_angles=10):
    lkf_stats = defaultdict(list)
    for ld, ldate in zip(lkfs_defs, dates):
        if ld is None:
            continue
        lkfs, defs = ld
        try:
            dil_angles, _ = get_lkf_intersection_angs(lkfs, defs, min_vort_significance=0.5)
        except:
            print('Cannot run get_lkf_intersection_angs')
            continue
        if len(dil_angles) < min_num_angles:
            continue
        kernel = gaussian_kde(dil_angles)
        bins = np.linspace(0, 180, 1 + 180//5)
        counts = kernel(bins)
        avg_angle = bins[np.argmax(counts)]
        lkf_lengths = [len(l[0]) for l in lkfs]
        avg_length = np.mean(lkf_lengths)

        lkf_stats['dates'].append(ldate)
        lkf_stats['angles'].append(avg_angle)
        lkf_stats['lengths'].append(avg_length)
        lkf_stats['counts'].append(len(lkfs))

    return lkf_stats

def upsample_lkf_stats(lkf_stats, dst_dates):
    dst_values = {}
    for key in lkf_stats:
        if key == 'dates':
            continue
        df = pd.DataFrame({'values': lkf_stats[key]}, index=lkf_stats['dates'])
        dfi = df.reindex(dst_dates).interpolate(method='linear')
        dst_values[key] = dfi.values
    return dst_values

def remove_negative_elements(x, y, t):
    xa, xb, xc = x[t].T
    ya, yb, yc = y[t].T
    jac = jacobian(xa, ya, xb, yb, xc, yc)
    if jac.min() > 0:
        return t

    t1 = np.array(t)
    for negi in np.unique(t[np.where(jac <=0)[0]]):
        gpi = np.all(t1 != negi, axis=1)
        t1 = t1[gpi]
    return remove_negative_elements(x, y, t1)
