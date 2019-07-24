from __future__ import absolute_import

import libsakurapy


def grid(working_set, gridfunction, convsupport, convsampling, weight_only,
         grid_real, grid_imag, wgrid_real, wgrid_imag, wsum_real, wsum_imag):
    #print 'LOG: sakura gridding function is called for real'
    #print 'LOG: sakura gridding function is called for imag'
    # grid real part
    libsakurapy.grid_convolving(
        working_set.nrow,
        0,
        working_set.nrow,
        #todo: check row_flag is consistent with sakura definition
        working_set.row_flag,
        working_set.u,
        working_set.v,
        convsupport,
        convsampling,
        working_set.npol,
        working_set.pol_map,
        working_set.nchan,
        working_set.channel_map,
        #todo: check flag is consistent with sakura definition
        working_set.flag,
        working_set.rdata,
        working_set.weight,
        False,
        len(gridfunction),
        gridfunction,
        grid_real.shape[2],  # npol_out
        grid_real.shape[3],  # nchan_out
        grid_real.shape[1],  # nh
        grid_real.shape[0],  # nv
        wsum_real,
        wgrid_real,
        grid_real)

    # grid imaginary part
    libsakurapy.grid_convolving(
        working_set.nrow,
        0,
        working_set.nrow,
        #todo: check row_flag is consistent with sakura definition
        working_set.row_flag,
        working_set.u,
        working_set.v,
        convsupport,
        convsampling,
        working_set.npol,
        working_set.pol_map,
        working_set.nchan,
        working_set.channel_map,
        #todo: check flag is consistent with sakura definition
        working_set.flag,
        working_set.idata,
        working_set.weight,
        False,
        len(gridfunction),
        gridfunction,
        grid_imag.shape[2],  # npol_out
        grid_imag.shape[3],  # nchan_out
        grid_imag.shape[1],  # nh
        grid_imag.shape[0],  # nv
        wsum_imag,
        wgrid_imag,
        grid_imag)


def solvemfista(l1, ltsqv, grid_data, image_data):
    grid_real = grid_data.real
    grid_imag = grid_data.imag
    print('LOG: solvemfista(l1, ltsqv, grid_real, grid_imag, image_data)')
