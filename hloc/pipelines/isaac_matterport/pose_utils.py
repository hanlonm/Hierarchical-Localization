import numpy as np
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt


def logmap_so3(R):
    """Logmap at the identity.
    Returns canonical coordinates of rotation.
    cfo, 2015/08/13

    """
    R11 = R[0, 0]
    R12 = R[0, 1]
    R13 = R[0, 2]
    R21 = R[1, 0]
    R22 = R[1, 1]
    R23 = R[1, 2]
    R31 = R[2, 0]
    R32 = R[2, 1]
    R33 = R[2, 2]
    tr = np.trace(R)
    omega = np.empty((3, ), dtype=np.float64)

    # when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, we do something
    # special
    if (np.abs(tr + 1.0) < 1e-10):
        if (np.abs(R33 + 1.0) > 1e-10):
            omega = (np.pi / np.sqrt(2.0 + 2.0 * R33)) * \
                np.array([R13, R23, 1.0+R33])
        elif (np.abs(R22 + 1.0) > 1e-10):
            omega = (np.pi / np.sqrt(2.0 + 2.0 * R22)) * \
                np.array([R12, 1.0+R22, R32])
        else:
            omega = (np.pi / np.sqrt(2.0 + 2.0 * R11)) * \
                np.array([1.0+R11, R21, R31])
    else:
        magnitude = 1.0
        tr_3 = tr - 3.0
        if tr_3 < -1e-7:
            theta = np.arccos((tr - 1.0) / 2.0)
            magnitude = theta / (2.0 * np.sin(theta))
        else:
            # when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
            # use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
            magnitude = 0.5 - tr_3 * tr_3 / 12.0

        omega = magnitude * np.array([R32 - R23, R13 - R31, R21 - R12])

    return omega


def compute_absolute_pose_error(p_es_aligned, q_es_aligned, p_gt, q_gt):
    e_trans_vec = (p_gt - p_es_aligned)
    e_trans = np.sqrt(np.sum(e_trans_vec**2, 1))

    # orientation error
    e_rot = np.zeros((len(e_trans, )))
    e_ypr = np.zeros(np.shape(p_es_aligned))
    for i in range(np.shape(p_es_aligned)[0]):
        R_we = pr.matrix_from_quaternion(q_es_aligned[i, :])
        R_wg = pr.matrix_from_quaternion(q_gt[i, :])
        e_R = np.dot(R_we, np.linalg.inv(R_wg))
        e_ypr[i, :] = pr.euler_from_matrix(e_R, 2, 1, 0, False)
        e_rot[i] = np.rad2deg(np.linalg.norm(logmap_so3(e_R[:3, :3])))
        pt.transform_log_from_transform

    # scale drift
    motion_gt = np.diff(p_gt, 0)
    motion_es = np.diff(p_es_aligned, 0)
    dist_gt = np.sqrt(np.sum(np.multiply(motion_gt, motion_gt), 1))
    dist_es = np.sqrt(np.sum(np.multiply(motion_es, motion_es), 1))
    e_scale_perc = np.abs((np.divide(dist_es, dist_gt) - 1.0) * 100)

    return e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc