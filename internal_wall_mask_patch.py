# internal_wall_mask_patch.py
# Straight internal wall: from (R/2, 0) to (R, 4) in (r,z).
# Builds masks and masked operators for a staggered (r,z) grid.

from __future__ import annotations
import numpy as np
from bc_segments_patch import (
    apply_boundary_normal_segmented,
    apply_ut_boundary_segmented,
    poisson_rhs_with_bc_segmented,
    integrate_wall_inflow_rR,
    make_bottom_drain_profile_auto,
)

# ----------------------------
#  A. Geometry: the wall line
# ----------------------------
def r_wall_of_z(R: float, z):
    """r_w(z) = R/2 + (R/8) * z for 0<=z<=4; inactive outside this band."""
    z = np.asarray(z)
    rw = (R/2.0) + (R/8.0) * z
    return rw

# ----------------------------
#  B. Build masks on a grid
# ----------------------------
def build_slanted_wall_masks(grid, R: float, z_min=0.0, z_max=4.0):
    """
    grid must expose:
      r_edges: (Nr+1,), z_edges: (Nz+1,), r_c: (Nr,), z_c: (Nz,)
      A_e, A_w, A_n, A_s: (Nr,Nz), V: (Nr,Nz)
    Returns dict with:
      solid_cell: (Nr,Nz) boolean (True = solid)
      open_e, open_w, open_n, open_s: (Nr,Nz) booleans for face connectivity (True = open fluid-fluid)
      R_line: float (outer radius) for convenience
    """
    r_c, z_c = grid.r_c, grid.z_c
    Nr, Nz = r_c.size, z_c.size
    Rline = grid.r_edges[-1]

    # Where the wall is active in z
    z_active = (z_c >= z_min) & (z_c <= z_max)
    r_wall = r_wall_of_z(R, z_c)  # (Nz,)

    # Cell is SOLID if its center lies to the "right" of wall (r >= r_wall) within active band
    solid_cell = np.zeros((Nr, Nz), dtype=bool)
    for j in range(Nz):
        if z_active[j]:
            solid_cell[:, j] = (r_c >= r_wall[j])
        else:
            solid_cell[:, j] = False

    # ---- Face "open" masks with face shapes ----
    # cell-centered solid map is 'solid_cell' (Nr, Nz)
    # radial faces: shape (Nr+1, Nz); interior i=1..Nr-1 is open iff both adjacent cells are fluid
    face_open_r = np.ones((Nr+1, Nz), dtype=bool)
    face_open_r[1:Nr, :] = ~(solid_cell[:-1, :] | solid_cell[1:, :])  # interior faces
    # boundaries (i=0 axis, i=Nr outer) remain True; handled by BCs

    # axial faces: shape (Nr, Nz+1); interior j=1..Nz-1 is open iff both adjacent cells are fluid
    face_open_z = np.ones((Nr, Nz+1), dtype=bool)
    face_open_z[:, 1:Nz] = ~(solid_cell[:, :-1] | solid_cell[:, 1:])  # interior faces
    # boundaries (j=0 bottom, j=Nz top) remain True; handled by BCs

    # Optional cell-centered “open” flags for convenience (not required for faces now)
    open_e = np.ones((Nr, Nz), dtype=bool); open_w = np.ones((Nr, Nz), dtype=bool)
    open_n = np.ones((Nr, Nz), dtype=bool); open_s = np.ones((Nr, Nz), dtype=bool)
    open_e[:-1, :] = ~(solid_cell[:-1, :] | solid_cell[1:, :])
    open_w[1:,  :] = ~(solid_cell[:-1, :] | solid_cell[1:, :])
    open_n[:, :-1] = ~(solid_cell[:, :-1] | solid_cell[:, 1:])
    open_s[:, 1: ] = ~(solid_cell[:, :-1] | solid_cell[:, 1:])

    masks = {
        "solid_cell": solid_cell,
        "face_open_r": face_open_r, "face_open_z": face_open_z,
        "open_e": open_e, "open_w": open_w, "open_n": open_n, "open_s": open_s,
        "R_line": Rline,
    }
    return masks

# -------------------------------------------------------
#  C. Enforce internal-wall BCs on predictor/corrected u
# -------------------------------------------------------
def apply_internal_wall_bc(u_r, u_z, u_t, grid, masks):
    """
    No-penetration on the internal wall: zero normal velocities on *closed* faces.
    Simple no-slip proxy for swirl: zero u_t in cells that touch any closed face.
    """
    face_open_r = masks["face_open_r"]  # (Nr+1, Nz)
    face_open_z = masks["face_open_z"]  # (Nr,   Nz+1)

    # Zero normal velocity on closed faces (keep boundaries handled by BCs as True in masks)
    u_r[:] = np.where(face_open_r, u_r, 0.0)
    u_z[:] = np.where(face_open_z, u_z, 0.0)

    # This is TOO AGGRESSIVE
    # # Swirl: zero in cells that touch any blocked face (cheap no-slip proxy)
    # if u_t is not None:
    #     touch_block = (~masks["open_e"]) | (~masks["open_w"]) | (~masks["open_n"]) | (~masks["open_s"])
    #     u_t[:] = np.where(touch_block, 0.0, u_t)

# -------------------------------------------------------
#  D. Masked divergence and Poisson RHS
# -------------------------------------------------------
def divergence_from_face_fluxes_masked(u_r, u_z, grid, masks):
    Ae, Aw, An, As, V = grid.A_e, grid.A_w, grid.A_n, grid.A_s, grid.V
    fr = masks["face_open_r"]  # (Nr+1, Nz)
    fz = masks["face_open_z"]  # (Nr,   Nz+1)

    # Zero contributions from closed faces by masking velocities (or areas)
    ur_e = u_r[1:, :]  * fr[1:, :]   # east faces (Nr, Nz)
    ur_w = u_r[:-1, :] * fr[:-1, :]  # west faces (Nr, Nz)
    uz_n = u_z[:, 1:]  * fz[:, 1:]   # north faces (Nr, Nz)
    uz_s = u_z[:, :-1] * fz[:, :-1]  # south faces (Nr, Nz)

    term_r = ur_e * Ae - ur_w * Aw
    term_z = uz_n * An - uz_s * As

    div = (term_r + term_z) / V
    div[masks["solid_cell"]] = 0.0
    return div

def poisson_rhs_with_bc_segmented_masked(grid, u_r_star, u_z_star, dt, rho, bc, masks,
                                         u_bottom=None,
                                         rhs_builder_no_mask=None):
    """
    Build pressure-correction RHS with both:
      - segmented external BCs (your previous bc_segments_patch logic), and
      - internal-wall masks (this module) that zero blocked-face terms & ignore solids.

    If you pass rhs_builder_no_mask=poisson_rhs_with_bc_segmented (from your bc patch),
    we will:
      1) call it to incorporate external BC terms,
      2) then re-zero blocked-face contributions and solid cells consistently.

    If rhs_builder_no_mask is None, we assemble a minimal masked RHS using only masks
    (no external segmented BCs).
    """
    if rhs_builder_no_mask is not None:
        # 1) Start with your segmented-BC RHS (external boundaries included)
        rhs_ext = poisson_rhs_with_bc_segmented(grid, u_r_star, u_z_star, dt, rho, bc, u_bottom=u_bottom)
        rhs_ext = rhs_ext.reshape(grid.Nr, grid.Nz, order='F')

        # 2) Replace the interior divergence with the masked one (respects internal wall)
        rhs_masked_div = - divergence_from_face_fluxes_masked(u_r_star, u_z_star, grid, masks) / dt

        # 3) Combine: use rhs_masked_div everywhere, then add *only* the external BC additive rows/cols.
        # Simple robust approach: take rhs_masked_div, then add (rhs_ext - rhs_unmasked_div) contributions.
        rhs_unmasked_div = - ((u_r_star[1:,:]*grid.A_e - u_r_star[:-1,:]*grid.A_w +
                               u_z_star[:,1:]*grid.A_n - u_z_star[:,:-1]*grid.A_s) / grid.V) / dt
        
        rhs = rhs_masked_div + (rhs_ext - rhs_unmasked_div)
    else:
        # Minimal: -(1/dt) div_masked(u*)
        rhs = -divergence_from_face_fluxes_masked(u_r_star, u_z_star, grid, masks) / dt

    # Now enforce internal-wall masking on RHS:
    rhs[masks["solid_cell"]] = 0.0

    return rhs.ravel(order='F')


